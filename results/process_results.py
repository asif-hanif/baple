import json
import os
import argparse



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='')
    parser.add_argument('--datasets', type=str, default='')
    # parser.add_argument('--method', type=str, default='')
    parser.add_argument('--seed', type=int, default=1)
    args = parser.parse_args()

    json_file = os.path.join( os.getcwd(), "results", "json", f"{args.model}_{args.datasets}_baple.json")
    
    if not os.path.exists(json_file):
        raise FileNotFoundError(f"File {json_file} not found. Please make sure the experiment has been run and the file exists.")

    with open(json_file, 'r') as f:
        results = json.load(f)

    clean_accuracy_list = []
    clean_f1_score_list = []

    backdoor_accuracy_list = []
    backdoor_f1_score_list = []

    target_classes = list(results['target_class'].keys())

    for target_class in target_classes:
        clean_accuracy_list.append(results['target_class'][target_class][f"seed_{args.seed}"]['clean']['acc'])
        clean_f1_score_list.append(results['target_class'][target_class][f"seed_{args.seed}"]['clean']['macro-f1'])

        backdoor_accuracy_list.append(results['target_class'][target_class][f"seed_{args.seed}"]['backdoor']['acc'])
        backdoor_f1_score_list.append(results['target_class'][target_class][f"seed_{args.seed}"]['backdoor']['macro-f1'])

    
    clean_avg_accuracy = sum(clean_accuracy_list) / len(clean_accuracy_list)
    clean_avg_f1_score = sum(clean_f1_score_list) / len(clean_f1_score_list)

    backdoor_accuracy_list = sum(backdoor_accuracy_list) / len(backdoor_accuracy_list)
    backdoor_f1_score_list = sum(backdoor_f1_score_list) / len(backdoor_f1_score_list)


    print(f"\n\n\n\nRESULTS (average across all target classes)\n")
    print(f"MODEL: {args.model.upper()}    DATASET: {args.datasets.upper()}    SEED: {args.seed}\n")
    print("#############################################################################\n")
    print("-------------------------------------   -------------------------------------")
    print("                 CLEAN                                 BACKDOOR           ")
    print("-------------------------------------   -------------------------------------")
    print(f"Accuracy= {clean_avg_accuracy:0.3f}      F-1 Score= {clean_avg_f1_score:0.3f}   Accuracy= {backdoor_accuracy_list:0.3f}      F-1 Score= {backdoor_f1_score_list:0.3f}")
    print("-------------------------------------   -------------------------------------")
    print("\n#############################################################################")
    print("\n\n")
        
