#script will brute force train each model and then run reccomendations on dev data.
import os

def main():

    flist = ['10', '25', '50', '70']
    reg = ['1.0', '0.1' ,'.01', '.001']
    epoch = ['1', '3', '5', '7']

    for j in epoch: 
        for i in reg:
            for k in flist:
                os.system("python train-model.py --hybrid --logistic -n Tune" + j + i + k + " -e " + j + " -k " + k + " -r " + i)
                os.system("python recommend.py --dev --cuda Tune" + j + i + k)

if __name__ == '__main__':
    main()