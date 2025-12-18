# llm_name: e5 lr 0.01
# llm_name: ST lr 0.1
# llm_name: llama lr 0.001 


for method in llama2_ood
do
    for dataset in cora
    do
        python main.py --dataset $dataset\
                        --ood_classifier $method
    done
done
