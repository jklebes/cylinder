import copy
import math
import random
import scipy.stats as stats

class Metropolis():
    def __init__(self, temperature, sigmas_init):
        self.temperature=temperature
        self.sigmas_init = sigmas_init
        self.sigmas = copy.copy(sigmas_init)
        self.sigmas_names = self.sigmas.keys()
        self.step_counter=0

            #dynamic step size
        self.acceptance_counter=0
        self.step_counter=0
        self.bump_step_counter=0
        self.target_acceptance = .5 #TODO: hard code as fct of nuber of parameter space dims
        self.ppf_alpha = -1 * stats.norm.ppf(self.target_acceptance / 2)
        self.m = 1
        self.ratio = ( #  a constant - no need to recalculate 
            (1 - (1 / self.m)) * math.sqrt(2 * math.pi) * math.exp(self.ppf_alpha ** 2 / 2) / 2 * self.ppf_alpha + 1 / (
            self.m * self.target_acceptance * (1 - self.target_acceptance)))
        self.sampling_width=.005
        self.bump_sampling_width=.005

    def metropolis_decision(self, old_energy, proposed_energy):
        """
        Considering energy difference and temperature, return decision to accept or reject step
        :param old_energy: current system total energy
        :param proposed_energy: system total energy after proposed change
        :return: bool True if step will be accepted; False to reject
        """
        diff = proposed_energy - old_energy
        #print("diff", diff, "temp", self.temp)
        assert (self.temperature is not None)
        if diff <= 0:
            return True  # choice was made that 0 difference -> accept change
        elif diff > 0 and self.temperature == 0:
            return False
        else:
            probability = math.exp(- 1 * diff / self.temperature)
            #print("probability", probability)
            if random.uniform(0, 1) <= probability:
                return True
            else:
                return False

    def update_sigma(self, accept, name):
        """
        TODO move to metropolis
        """
        self.step_counter+=1
        step_number_factor = max((self.step_counter / self.m, 200))
        steplength_c = self.sigmas[name] * self.ratio
        if accept:
            self.sigmas[name] += steplength_c * (1 - self.target_acceptance) / step_number_factor
        else:
            self.sigmas[name] -= steplength_c * self.target_acceptance / step_number_factor
        assert (self.sigmas[name]) > 0
        #print(self.step_counter)
        #print("acceptance" , self.acceptance_counter, self.acceptance_counter/self.step_counter,self.sampling_width)