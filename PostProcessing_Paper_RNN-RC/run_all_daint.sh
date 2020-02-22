
module load daint-gpu
module load cray-python/3.6.5.7
source ${HOME}/venv-3.6-pytorch/bin/activate

Experiment_Name="Experiment_Daint_Large"

for system_name in "Lorenz96_F8GP40R40" 
# for system_name in "Lorenz96_F8GP40R40" "Lorenz96_F10GP40R40" 
# for system_name in "Lorenz96_F8GP40R40" "Lorenz96_F10GP40R40" "Lorenz96_F8GP40"
# for system_name in "Lorenz96_F8GP40"
do
    echo ${system_name}
    # for i in 0 1 2 3 4 5 6 7 8 9 10 10c 11 12 12b 12c 13 15
    for i in 12c
    do
        echo "RUNING SYSTEM ${system_name}, RUNNING SCRIPT ${i}"
        python3 POSTPROCESS_${i}.py --system_name ${system_name} --Experiment_Name ${Experiment_Name}
    done
done
