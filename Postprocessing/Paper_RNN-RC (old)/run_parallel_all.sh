


Experiment_Name="Experiment_Daint_Large"

for system_name in "KuramotoSivashinskyGP512" "Lorenz96_F8GP40"
# for system_name in "Lorenz96_F8GP40"
do
    echo ${system_name}
    python3 POSTPROCESS_0.py --system_name ${system_name} --Experiment_Name ${Experiment_Name}
    for i in 2 10 10c 11 12 13 13b 14 15
    # for i in 10
    do
        echo "RUNING SYSTEM ${system_name}, RUNNING SCRIPT ${i}"
        python3 POSTPROCESS_PARALLEL_${i}.py --system_name ${system_name} --Experiment_Name ${Experiment_Name}
    done
done

