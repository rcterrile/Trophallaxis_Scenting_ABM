from mesa.model import Model
from bee_troph.model import TrophallaxisABM
import numpy as np

def _get_model_parameters(n: int=110, frac_fed: int=10, th: int=180,
                          # aboost: int=10, epsboost: int=8, endboost: int=8,
                          dec_rate=7.8, Dc=0.6, del_t=0.005, trans_prob=0.5,
                          A_s=0.575, wb_s=10.0, thresh_s=0.3, scent_prob=0.3,
                          ses_i=0, b_i=0, r_i=0):
    model_params = {
        "N": n,
        "fraction_of_fed_bees": frac_fed,
        "theta": th,
        # "solid_bounds": True,
        "data_out": True,
        "session_id": ses_i,
        "batch_id": b_i,
        "run_id": r_i,
        "scent_thresh": 0.15,   # threshold required for fed bee to emit
        "scent_prob": scent_prob,      # probability for fed bee to emit
        "scent_freq": 80,       # emission frequency of fed bee
        "scent_move": True,         # whether fed bees can move while emitting
        # "a_boost": aboost,
        # "eps_boost": epsboost,
        # "end_boost": endboost,
        "dec_rate": dec_rate,
        "Dc": Dc,
        "del_t": del_t,
        "wb_": wb_s,
        "A_": A_s,
        "fan_threshold": thresh_s,      # threshold for bee to fan wings and scent
        "trans_prob_": trans_prob,
        "width": 32,
        "height": 32,
    }
    return model_params
# _get_model_parameters()

def batch_run(session_num: int = 0, batch_num: int = 0, repetitions: int = 10,
                r_start: int=0, max_steps: int = 1000, display_progress: bool = True,
                # atb: int = 1, epb: int=8, enb: int=1,
                dec_rate: float=7.9, Dc: float=0.6, del_t: float=0.005,
                trans_prob: float=0.5, A_s: float=0.575, wb_s: float=40.0,
                thresh_s: float=0.3, scent_prob: float=0.3, n: int = 110,
                frac_fed: int = 10, th: int = 180):
    ## Start terminal printout:
    if display_progress:
        print("BATCH: " + str(batch_num))

    ## Get parameters:
    # params = _get_model_parameters(n, frac_fed, attr_rad, th, atb, epb, enb)
    params = _get_model_parameters(n, frac_fed, th, dec_rate, Dc, del_t, trans_prob, A_s, wb_s, thresh_s, scent_prob, session_num, batch_num, r_start)
    # params["session_id"] = session_num
    # params["batch_id"] = batch_num
    ## Run model and repeat 'repetitions' times:
    id_counter = r_start                #  ---- change if not starting at run 0 ----
    for i in range(repetitions):
        params["run_id"] = i + id_counter
        # if display_progress:
        _run_model(TrophallaxisABM, params, max_steps)
        # if display_progress:
        # if display_progress:
        #     if i%5 == 0:
        #         # print("- " + str(i+1) + " of " + str(repetitions))
        #         print(str(int((i+1)/repetitions*100))+"%")
    if display_progress:
        print("Batch Finished")
# batch_run()# # # # # # # # # #

def _run_model(model_cls: type[Model], model_params, max_steps: int): #, iter: int, rep: int, max_reps: int):
    print("|" + str(model_params['run_id']), end="|")
    model = model_cls(**model_params)
    while model.running and model.schedule.steps <= max_steps:
        model.step()
        if model.t_i%30 == 0:
            print("#",end="")
    if model.running:
        print("oops")
    print("|")
# _run_model()

# batch_run(sess, batch, reps, start_rep, max_x, display,
#           decay_rate, Dc, del_t, trans_prob,
#           A_s, wb_, thresh_s,
#           n, frac_fed, attr_rad, theta)

##############[RUNS TO PELEG SERVER]##############
### -SESSION 0- ###
batch_run(0, 36, 20, 0, 600, True,        ## 36
            7.8, 60.0, 0.005, 0.5,       # Dc=60, decay=7.8, trans_prob=0.5
            0.575, 40.0, 0.5, 0.3)       # A=0.575, wb=40, thresh=0.5, sc_prob=0.3

# #### NO SCENTING RUN ####
# batch_run(0, 12, 20, 0, 600, True,        ## 8
#             7.8, 60.0, 0.005, 0.0,       # Dc=60, decay=7.8, trans_prob=0.5
#             0.0, 0.0, 200)                 # A=0.575, wb=0, threshold=0.20
