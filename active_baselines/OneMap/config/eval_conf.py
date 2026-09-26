from spock import spock, SpockBuilder

from config import HabitatControllerConf, MappingConf, PlanningConf


@spock
class EvalConf:
    multi_object: bool
    max_steps: int
    max_dist: float
    log_rerun: bool
    is_gibson: bool
    is_hssd: bool                   #TODO Changed
    is_trial: bool                  #TODO Changed
    controller: HabitatControllerConf
    mapping: MappingConf
    planner: PlanningConf
    object_nav_path: str
    scene_path: str
    use_pointnav: bool
    square_im: bool

    use_saved_steps: bool   #TODO Added
    saved_steps_dir: str    #TODO Changed: Directory for saved steps navigation
    results_dir: str
    run_split: int


def load_eval_config():
    return SpockBuilder(EvalConf, HabitatControllerConf, PlanningConf, MappingConf,
                        desc='Default MON config.').generate()
