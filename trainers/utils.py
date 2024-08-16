import os
import json



def save_results(cfg, results, var_name, var_value):

    # save results in a json file
    results_file = os.path.join("results", "json", f"{cfg.MODEL_NAME}_{cfg.DATASET_NAME}_baple.json")
    
    if not os.path.exists(results_file):
        # create the file if it doesn't exist
        with open(results_file, "w") as file:
            file.write("{}")
            
    # load existing results
    with open(results_file, "r") as file:
        results_json = json.load(file)
        

    if var_name not in results_json:
        results_json[var_name] = {}


    if var_value not in results_json[var_name]:
        results_json[var_name][var_value] = {}
    
    
    results_json[var_name][var_value] = {
        **results_json[var_name][var_value],  
        **{f"seed_{cfg.SEED}" : { "clean":    { "acc": float(f"{results[0]['accuracy']/100:0.4f}"), "macro-f1": float(f"{results[0]['macro_f1']/100:0.4f}") },
                                  "backdoor": { "acc": float(f"{results[1]['accuracy']/100:0.4f}"), "macro-f1": float(f"{results[1]['macro_f1']/100:0.4f}") }
                                }
          }
        }

    # save configs
    if "configs" not in results_json:
        results_json["configs"] = cfg


    # save updated results
    with open(results_file, "w") as file:
        json.dump(results_json, file, indent=2)





def print_results(cfg, results):
    print(f"\n\n\n\nRESULTS  (MODEL: {cfg.MODEL_NAME.upper()}   DATASET: {cfg.DATASET_NAME.upper()}   TARGET CLASS: {cfg.BACKDOOR.TARGET_CLASS})\n")
    print("#############################################################################")
    print("-------------------------------------   -------------------------------------")
    print("                 CLEAN                                 BACKDOOR           ")
    print("-------------------------------------   -------------------------------------")
    print(f"Accuracy = {results[0]['accuracy']/100:0.3f}    F-1 Score = {results[0]['macro_f1']/100:0.3f}   Accuracy = {results[1]['accuracy']/100:0.3f}    F-1 Score = {results[1]['macro_f1']/100:0.3f}")
    print("-------------------------------------   -------------------------------------")
    print("\n#############################################################################")
    print("\n\n")