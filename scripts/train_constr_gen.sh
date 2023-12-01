# ball
python train_constr_gen.py --constraint ball --xdim 2 --p0 moon    --name moon
python train_constr_gen.py --constraint ball --xdim 2 --p0 spiral  --name spiral
python train_constr_gen.py --constraint ball --xdim 2 --p0 gmm     --name gmm
python train_constr_gen.py --constraint ball --xdim 6 --p0 gmm_nd  --name gmm6d
python train_constr_gen.py --constraint ball --xdim 8 --p0 gmm_nd  --name gmm8d
python train_constr_gen.py --constraint ball --xdim 20 --p0 gmm_nd --name gmm20d --num-itr 100000

# simplex
python train_constr_gen.py --constraint simplex --xdim 2  --p0 dirichlet1   --name simplex3d
python train_constr_gen.py --constraint simplex --xdim 2  --p0 dirichlet2   --name simplex3dv2
python train_constr_gen.py --constraint simplex --xdim 6  --p0 dirichlet3   --name simplex7d
python train_constr_gen.py --constraint simplex --xdim 8  --p0 dirichlet4   --name simplex9d
python train_constr_gen.py --constraint simplex --xdim 19 --p0 dirichlet_nd --name simplex20d --num-itr 100000

# cube
python train_constr_gen.py --constraint cube --xdim 2 --p0 cube --name cube2d
python train_constr_gen.py --constraint cube --xdim 3 --p0 cube --name cube3d
python train_constr_gen.py --constraint cube --xdim 6 --p0 cube --name cube6d
python train_constr_gen.py --constraint cube --xdim 8 --p0 cube --name cube8d
python train_constr_gen.py --constraint cube --xdim 20 --p0 cube --name cube20d --num-itr 100000
