## About
This project is the reconstruction and reproduction of the logistic model, which reduces the three parameters of the logistic model to two parameters by methods such as inference and normalization, reducing the size of the solution space and achieving better model performance. At the same time, the project reproduces the original logistics model for performance comparison, and the dataset is mainly based on simulation data and COVID-19 epidemic data.
## Data
The COVID-19 data were obtained from the John Hopkins Coronavirus Resource Center(<a>https://coronavirus.jhu.edu/map.html</a>). The COVID-19 data were updated on a daily basis, while the influenza data were updated weekly.
## Method
### Simulated Data
The simulated data is mainly used to test the performance of single model fitting, and the noise data is added to simulate the interference of the actual situation in the real data, which effectively tests the anti-interference performance and model sensitivity of the model, as well as testing the model running time in a controlled situation.\
### Epidemic Data
In the `COVID-19 epidemic data`, a prediction model optimization algorithm for sigmoid-like functions is implemented, and a suitable fitting model is selected by combining the information criterion with the push-pull and scaling of the time window to optimize the prediction effect of the model in practical use. The optimization method is applied to the original logistics model and the two-parameter logistics model to construct the prediction algorithm, and through the comparison of the effect, the prediction effect is better than that of the complex model, and the running time is shorter.
## Project Structure
main_pro --- The main project, which includes each model implementation and real dataset, as well as model usage.\
simulatedData --- Generation of the simulated data set and performance of the original logistics model and the two-parameter logistics model on the simulated data set.\
resultFile --- Partial run result data for the project, as well as test data for the model in other domains.\
predict_true_img --- Visualization of some of the project's run results.\
Reference Models --- Reference results data for the reference models used by the project(LANL, Delphi, SIKJalpha).\
