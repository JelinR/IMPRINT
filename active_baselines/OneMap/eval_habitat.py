from eval.habitat_evaluator import HabitatEvaluator
from config import load_eval_config
from eval.actor import MONActor
from eval.actor_saved_nav import MONActor as MONActor_saved

def main():
    # Load the evaluation configuration
    eval_config = load_eval_config()
    
    #TODO Changed: Printing Eval Config Args
    print(eval_config)
    use_saved_steps = eval_config.EvalConf.use_saved_steps

    # Create the HabitatEvaluator object
    if use_saved_steps:
        print(f"\n\n ---- Using Saved Navigation! -----\n\n")
        evaluator = HabitatEvaluator(eval_config.EvalConf, MONActor_saved(eval_config.EvalConf))
    else:
        evaluator = HabitatEvaluator(eval_config.EvalConf, MONActor(eval_config.EvalConf))
    
    evaluator.evaluate()

if __name__ == "__main__":
    main()
