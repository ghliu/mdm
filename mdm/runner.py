import pickle
import functools

import torch
from torch.optim import AdamW, lr_scheduler
from torch_ema import ExponentialMovingAverage

from guided_diffusion import script_util as gd_util
from .network import ToyPolicy
from .metrices import compute_metrics
from .plotting import save_snapshot
from .writer import build_log_writer

from ipdb import set_trace as debug

def build_optimizer_sched(opt, net, log):

    optim_dict = {"lr": opt.lr, 'weight_decay': opt.l2_norm}
    optimizer = AdamW(net.parameters(), **optim_dict)
    log.info(f"[Opt] Built AdamW optimizer {optim_dict=}!")

    if opt.lr_gamma < 1.0:
        sched_dict = {"step_size": opt.lr_step, 'gamma': opt.lr_gamma}
        sched = lr_scheduler.StepLR(optimizer, **sched_dict)
        log.info(f"[Opt] Built lr step scheduler {sched_dict=}!")
    else:
        sched = None

    if opt.load:
        checkpoint = torch.load(opt.load, map_location="cpu")
        if "optimizer" in checkpoint.keys():
            optimizer.load_state_dict(checkpoint["optimizer"])
            log.info(f"[Opt] Loaded optimizer ckpt {opt.load}!")
        else:
            log.warning(f"[Opt] Ckpt {opt.load} has no optimizer!")
        if sched is not None and "sched" in checkpoint.keys() and checkpoint["sched"] is not None:
            sched.load_state_dict(checkpoint["sched"])
            log.info(f"[Opt] Loaded lr sched ckpt {opt.load}!")
        else:
            log.warning(f"[Opt] Ckpt {opt.load} has no lr sched!")

    return optimizer, sched

class Runner(object):
    def __init__(self, opt, log, save_opt=True):
        super(Runner,self).__init__()

        # Save opt.
        if save_opt:
            opt_pkl_path = opt.ckpt_path / "options.pkl"
            with open(opt_pkl_path, "wb") as f:
                pickle.dump(opt, f)
            log.info("Saved options pickle to {}!".format(opt_pkl_path))

        self.diffusion = gd_util.create_gaussian_diffusion(
            steps=opt.interval, noise_schedule=opt.noise_sched
        )
        log.info(f"[Diffusion] Built diffusion: steps={opt.interval}, schedle={opt.noise_sched}!")

        self.net = ToyPolicy(data_dim=opt.xdim)
        self.ema = ExponentialMovingAverage(self.net.parameters(), decay=opt.ema)
        if opt.load:
            checkpoint = torch.load(opt.load, map_location="cpu")
            self.net.load_state_dict(checkpoint['net'])
            log.info(f"[Net] Loaded network ckpt: {opt.load}!")
            self.ema.load_state_dict(checkpoint["ema"])
            log.info(f"[Ema] Loaded ema ckpt: {opt.load}!")

        self.net.to(opt.device)
        self.ema.to(opt.device)

        self.log = log

    def sample_batch(self, opt, sampler, constraint):
        x0 = sampler.sample()
        return constraint.to_dual(x0).to(opt.device)

    def train(self, opt, data_sampler, constraint):
        self.writer = build_log_writer(opt)
        log, net, ema = self.log, self.net, self.ema
        optimizer, sched = build_optimizer_sched(opt, net, log)

        save_snapshot(opt, f"{opt.ckpt_path}/p01.pdf", constraint, x0=data_sampler.sample())

        net.train()
        for it in range(1, opt.num_itr+1):
            optimizer.zero_grad()

            y0 = self.sample_batch(opt, data_sampler, constraint)
            if torch.isnan(y0).any():
                log.error(f"Error at {it=}!")
                continue
            step = torch.randint(0, opt.interval, (y0.shape[0],)).to(opt.device)

            compute_losses = functools.partial(
                self.diffusion.training_losses,
                self.net,
                y0,
                step,
            )
            losses = compute_losses()
            loss = losses["loss"].mean()
            if torch.isnan(loss):
                log.error(f"Error2 at {it=}!")
                continue
            loss.backward()

            optimizer.step()
            ema.update()
            if sched is not None: sched.step()

            # -------- logging --------
            if it % 100 == 0:
                log.info("train_it {}/{} | lr:{} | loss:{}".format(
                    1+it,
                    opt.num_itr,
                    "{:.2e}".format(optimizer.param_groups[0]['lr']),
                    "{:+.4f}".format(loss.item()),
                ))
                self.writer.add_scalar(it, 'loss', loss.detach())

            if it % opt.save_itr == 0:
                torch.save({
                    "net": self.net.state_dict(),
                    "ema": ema.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "sched": sched.state_dict() if sched is not None else sched,
                }, opt.ckpt_path / "latest.pt")
                log.info(f"Saved latest({it=}) checkpoint to {opt.ckpt_path=}!")

            if it % opt.eval_itr == 0:
                net.eval()
                self.evaluation(opt, it, data_sampler, constraint)

                net.train()
        self.writer.close()

    @torch.no_grad()
    def generate(self, opt, constraint):
        with self.ema.average_parameters():
            self.net.eval()
            y0 = self.diffusion.p_sample_loop(
                self.net,
                (opt.batch_size, opt.xdim),
                clip_denoised=False, # note: dual space shouldn't clip!
            )
        y0 = y0.detach().cpu()
        pred_x0 = constraint.to_primal(y0)

        return pred_x0, y0

    @torch.no_grad()
    def evaluation(self, opt, it, datasampler, constraint):

        log = self.log
        log.info(f"========== Evaluation started: iter={it} ==========")

        pred_x0, y0 = self.generate(opt, constraint)

        save_snapshot(opt, f"{opt.ckpt_path}/recon{str(it).zfill(5)}.pdf", constraint, y0=y0)

        # compute distribution metrics
        ref_x0 = datasampler.sample()
        metrices = compute_metrics(pred_x0, ref_x0)
        for k, v in metrices.items():
            log.info(f"{k}: {v:.4f}")
            self.writer.add_scalar(it, k, v)

        log.info(f"========== Evaluation finished: iter={it} ==========")
        torch.cuda.empty_cache()
