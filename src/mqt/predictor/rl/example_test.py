from predictor import Predictor

rl_pred = Predictor(
    figure_of_merit="expected_fidelity", device_name="ibm_washington"
)
rl_pred.train_model(timesteps=100000, model_name="sample_model_rl")