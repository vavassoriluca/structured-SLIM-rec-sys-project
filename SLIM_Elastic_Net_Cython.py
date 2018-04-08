#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 31/03/2018

@author: Alex Porciani & Luca Vavassori
"""


from Base.Recommender import Recommender

import subprocess
import os, sys, time

import numpy as np



def default_validation_function(self):


    return self.evaluateRecommendations(self.URM_validation)



class SLIM_Elastic_Net_Cython(Recommender):

    RECOMMENDER_NAME = "SLIM_Elastic_Net_Cython"


    def __init__(self, ICM, URM_train, positive_threshold=1, URM_validation = None,
                 recompile_cython = False, symmetric = True):


        super(SLIM_Elastic_Net_Cython, self).__init__()


        if URM_validation is not None:
            self.URM_validation = URM_validation.copy()
        else:
            self.URM_validation = None


        self.URM_mask = self.URM_train.copy()

        self.URM_mask.data = self.URM_mask.data >= self.positive_threshold
        self.URM_mask.eliminate_zeros()

        self.ICM = ICM

        if recompile_cython:
            print("Compiling in Cython")
            self.runCompilationScript()
            print("Compilation Complete")



    def fit(self, epochs=300, logFile=None, URM_test=None, batch_size = 1000, learning_rate = 1e-4, topK = 200,
            sgd_mode='adagrad', gamma=0.995, beta=0.9,
            stop_on_validation = False, lower_validatons_allowed = 5, validation_metric = "map",
            validation_function = None, validation_every_n = 1):


        # Import compiled module
        from SLIM_Elastic_Net.Cython.SLIM_Elastic_Net_Cython import SLIM_Elastic_Net_Cython

        # Select only positive interactions
        URM_train_positive = self.URM_train.copy()

        URM_train_positive.data = URM_train_positive.data >= self.positive_threshold
        URM_train_positive.eliminate_zeros()

        self.sgd_mode = sgd_mode
        self.epochs = epochs


        self.cythonEpoch = SLIM_Elastic_Net_Cython(self.ICM, self.URM_mask,
                                                   l=learning_rate,
                                                   b=beta,
                                                   g=gamma,
                                                   epochs=epochs)

        if (topK < 1):
            raise ValueError("TopK not valid. Acceptable values are either False or a positive integer value. Provided value was '{}'".format(topK))

        self.topK = topK

        self.cythonEpoch.fit(topK=topK,
                             shrink=0,
                             normalize=True,
                             mode="cosine")

        self.logFile = logFile

        if validation_every_n is not None:
            self.validation_every_n = validation_every_n
        else:
            self.validation_every_n = np.inf

        if validation_function is None:
            validation_function = default_validation_function


        self.batch_size = batch_size
        self.learning_rate = learning_rate


        start_time = time.time()


        best_validation_metric = None
        lower_validatons_count = 0
        convergence = False

        self.S_incremental = self.cythonEpoch.get_S()
        self.S_best = self.S_incremental.copy()
        self.epochs_best = 0

        currentEpoch = 0

        while currentEpoch < self.epochs and not convergence:

            if self.batch_size>0:
                self.cythonEpoch.epochIteration_Cython()
            else:
                print("No batch not available")

            # Determine whether a validaton step is required
            if self.URM_validation is not None and (currentEpoch + 1) % self.validation_every_n == 0:

                print("SLIM_BPR_Cython: Validation begins...")

                self.get_S_incremental_and_set_W()

                results_run = validation_function(self)

                print("SLIM_BPR_Cython: {}".format(results_run))

                # Update the D_best and V_best
                # If validation is required, check whether result is better
                if stop_on_validation:

                    current_metric_value = results_run[validation_metric]

                    if best_validation_metric is None or best_validation_metric < current_metric_value:

                        best_validation_metric = current_metric_value
                        self.S_best = self.S_incremental.copy()
                        self.epochs_best = currentEpoch +1

                    else:
                        lower_validatons_count += 1

                    if lower_validatons_count >= lower_validatons_allowed:
                        convergence = True
                        print("SLIM_BPR_Cython: Convergence reached! Terminating at epoch {}. Best value for '{}' at epoch {} is {:.4f}. Elapsed time {:.2f} min".format(
                            currentEpoch+1, validation_metric, self.epochs_best, best_validation_metric, (time.time() - start_time) / 60))


            # If no validation required, always keep the latest
            if not stop_on_validation:
                self.S_best = self.S_incremental.copy()

            print("SLIM_BPR_Cython: Epoch {} of {}. Elapsed time {:.2f} min".format(
                currentEpoch+1, self.epochs, (time.time() - start_time) / 60))

            currentEpoch += 1



        self.get_S_incremental_and_set_W()

        sys.stdout.flush()




    def writeCurrentConfig(self, currentEpoch, results_run, logFile):

        current_config = {'lambda_i': self.lambda_i,
                          'lambda_j': self.lambda_j,
                          'batch_size': self.batch_size,
                          'learn_rate': self.learning_rate,
                          'topK_similarity': self.topK,
                          'epoch': currentEpoch}

        print("Test case: {}\nResults {}\n".format(current_config, results_run))
        # print("Weights: {}\n".format(str(list(self.weights))))

        sys.stdout.flush()

        if (logFile != None):
            logFile.write("Test case: {}, Results {}\n".format(current_config, results_run))
            # logFile.write("Weights: {}\n".format(str(list(self.weights))))
            logFile.flush()



    def runCompilationScript(self):

        # Run compile script setting the working directory to ensure the compiled file are contained in the
        # appropriate subfolder and not the project root

        compiledModuleSubfolder = "/SLIM_Elastic_Net/Cython"
        #fileToCompile_list = ['Sparse_Matrix_CSR.pyx', 'SLIM_BPR_Cython_Epoch.pyx']
        fileToCompile_list = ['SLIM_Elastic_Net_Cython.pyx']

        for fileToCompile in fileToCompile_list:

            command = ['python',
                       'compileCython.py',
                       fileToCompile,
                       'build_ext',
                       '--inplace'
                       ]


            output = subprocess.check_output(' '.join(command), shell=True, cwd=os.getcwd() + compiledModuleSubfolder)

            try:

                command = ['cython',
                           fileToCompile,
                           '-a'
                           ]

                output = subprocess.check_output(' '.join(command), shell=True, cwd=os.getcwd() + compiledModuleSubfolder)

            except:
                pass


        print("Compiled module saved in subfolder: {}".format(compiledModuleSubfolder))

        # Command to run compilation script
        # python compileCython.py SLIM_BPR_Cython_Epoch.pyx build_ext --inplace

        # Command to generate html report
        # cython -a SLIM_BPR_Cython_Epoch.pyx


