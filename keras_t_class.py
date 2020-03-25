from kerastuner import HyperModel
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers  import Adam
from tensorflow.keras.losses import MeanAbsoluteError
from kerastuner import HyperModel,Tuner
from kerastuner.tuners import RandomSearch,Hyperband,BayesianOptimization

HYPERBAND_MAX_EPOCHS = 40
MAX_TRIALS = 20
EXECUTION_PER_TRIAL = 2
N_EPOCH_SEARCH = 40
BAYESIAN_NUM_INITIAL_POINTS = 1
SEED=1

class NNTSMLPModel(HyperModel):
    def __init__(self, input_shape=None, num_classes=None):
       # if input_shape == 0:
        self.input_shape = input_shape
        self.num_classes = num_classes
    
    def build(self, hp):
        
        model =Sequential()
        model.add(Dense(units=hp.Int(
                                    'units',
                                    min_value= 50,
                                    max_value=200,
                                    step=50,
                                    default=50
                                    ), 
                        activation=hp.Choice(
                                    'dense_activation',
                                    values=['relu', 'tanh', 'sigmoid'],
                                    default='relu'
                                    ),
                        input_dim= self.input_shape
                        )
                )
        for i in range(hp.Int('num_layers', 1, 6)):
                                            model.add(Dense(
                                                units=hp.Int('units_' + str(i),
                                                            min_value=5,
                                                            max_value=200,
                                                            step=20),
                                             
                                            activation=hp.Choice(
                                                        'dense_activation',
                                                        values=['relu', 'tanh', 'sigmoid'],
                                                        default='relu'
                                                        ),
                                            ))

                                            model.add(Dropout(rate=hp.Float(
                                                                            'dropout'+ str(i),
                                                                            min_value=0.0,
                                                                            max_value=0.5,
                                                                            default=0.25,
                                                                            step=0.05
                                                                        )
                                                            )
                                            )
        
        model.add(Dense(self.num_classes ,activation=hp.Choice('dense_activation_final',
                                                values=['relu', 'tanh', 'sigmoid'],
                                                default='sigmoid'
                                              )
                        )
                )
        
        optimizer = hp.Choice('optimizer',['adam','sgd'])
        model.compile(optimizer,loss=MeanAbsoluteError(), metrics=['mape'])
        return model

def define_tuners(hypermodel, directory, project_name):
    random_tuner = RandomSearch(
        hypermodel,
        objective="val_loss",
        seed=SEED,
        max_trials=MAX_TRIALS,
        executions_per_trial=EXECUTION_PER_TRIAL,
        directory=f"{directory}_random_search",
        project_name=project_name,
    )
    hyperband_tuner = Hyperband(
        hypermodel,
        max_epochs=HYPERBAND_MAX_EPOCHS,
        objective="val_loss",
        seed=SEED,
        executions_per_trial=EXECUTION_PER_TRIAL,
        directory=f"{directory}_hyperband",
        project_name=project_name,
    )
    bayesian_tuner = BayesianOptimization(
        hypermodel,
        objective='val_loss',
        seed=SEED,
        num_initial_points=BAYESIAN_NUM_INITIAL_POINTS,
        max_trials=MAX_TRIALS,
        directory=f"{directory}_bayesian",
        project_name=project_name
    )
    return [random_tuner, hyperband_tuner, bayesian_tuner]

class MyTunner(Tuner):
    def run_trial(self,trial,train_ds):
        hp= trial.hyperparameters
        train_ds = train_ds.batch(hp.Int('batch_size',5,20, step=5,default=15))
        lr= hp.float('learning_rate', 1e-4,1e-2,sampling='log', default=1e-3)
        optimizer = tf.keras.optimizers.Adam(lr)
        epoch_loss_metric = tf.keras.metrics.Mean_Absolute_percentage_error()
        model =  self.hypermodel.build(trial.hyperparameters)

        @tf.function
        def run_train_step(data):            
            # separate inputs and outputs
            x_train, y_train = data[:, :-1], data[:, -1]                 
            
            with tf.GradientTape() as tape:
                logits = model(x_train)
                loss = tf.keras.losses.MeanAbsoluteError(y_train,logits)
                if model.losses:
                    loss += tf.math.add_n(model.losses)
                gradients = tape.gradient(loss,model.trainable_variables)
            optimizer.apply_gradients(zip(gradients,model.trainable_variables))
            epoch_loss.metric.ipdate_state(loss)
            return loss
        
        for epoch in range(30):
            print ('Epoch: {}'.format(epoch))

            self.on_epoch_begin(trial,model,epoch,logs={})
            for bacth, data in enumerate(train_ds):
                self.on_batch_begin(trial,model,epoch,logs={})
                batch_loss = float(run_train_step(data))
                self.on_batch_end(trial,model,batch,logs={'loss': batch_loss})

                if batch %100==0:
                    loss = epoch_loss_metric.result().numpy()
                    print('Batch: {}, Average Loss: {}'.format(bacth,loss))
            
            epoch_loss = epoch_loss_metric.result().numpy()
            self.on_epoch_end(trial,model,epoch,logs={'loss': epoch_loss})
            epoch_loss_metric.reset_states()




