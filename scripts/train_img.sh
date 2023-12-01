DATASET=$1

echo "Note: available dataset are ffhq, afhqv2 "
echo "Specified [$DATASET]"

# these arguments should NOT be modified unless pretrain models change
AFHQv2="--pretrain=data/edm-afhqv2-64x64-uncond-vp.pt --cond=0 --arch=ddpmpp --batch=256 --cres=1,2,2,2 --lr=2e-4 --dropout=0.25 --augment=0.15"
FFHQ="  --pretrain=data/edm-ffhq-64x64-uncond-vp.pt   --cond=0 --arch=ddpmpp --batch=256 --cres=1,2,2,2 --lr=2e-4 --dropout=0.05 --augment=0.15"

# `batch-gpu` is the maximum batch per gpu. **Current values are tuend on TITAN RTX**
# `snap` controls the frequency (in kimg) of saving network snapshot
# `dump` controls the frequency (in kimg) of dumpping training stats

if [ "$DATASET" == afhqv2 ]; then
    python train_watermark.py --data=data/afhqv2-64x64.zip --outdir=results $AFHQv2 --batch-gpu=64 --snap=20 --dump=20
fi

if [ "$DATASET" == ffhq ]; then
    python train_watermark.py --data=data/ffhq-64x64.zip --outdir=results $FFHQ   --batch-gpu=64 --snap=10 --dump=10
fi
