from evaluate import get_env, get_state_action_size, evaluate
from policy import NeuroevoPolicy
from argparse import ArgumentParser
import numpy as np
import logging
from pymoo.operators.crossover import erx
from pymoo.factory import get_crossover

erc = get_crossover("perm_erx")

def oneplus_lambda(x, fitness, gens, lam, std=0.01, rng=np.random.default_rng()):
    x_best = x
    f_best = -np.Inf
    n_evals = 0
    for g in range(gens):
        N = rng.normal(size=(lam, len(x))) * std
        for i in range(lam):
            ind = x + N[i, :]
            f = fitness(ind)
            if f > f_best:
                f_best = f
                x_best = ind
        x = x_best
        n_evals += lam
        logging.info('\t%d\t%d', n_evals, f_best)
    return x_best


def evaluate_pop(population, fit):
    fitness_population = np.zeros(len(population))
    for i in range(len(population)):
        fitness_population[i] = fit(population[i])
    return fitness_population


def fp_selection(population, fitness_population):
    p = (np.min(fitness_population) - fitness_population)
    if len(np.unique(p)) == 1:
        p = np.ones(len(population))
    p /= np.sum(p)
    rng = np.random.default_rng()
    ind = rng.choice(len(population), p=p, size=int(len(population)*0.4))
    return population[ind], fitness_population[ind]


def truncation_selection(population, fitness_population, p=0.2):
    n_elites = int(np.floor(len(population) * p))
    elites = np.argsort(fitness_population)[-n_elites:]
    return population[elites], fitness_population[elites]


def tournament_selection(population, fitness_population, t_size=2):
    inds = rng.choice(len(population), t_size)
    ind = inds[np.argmax(fitness_population[inds])]
    return population[ind], fitness_population[ind]


def mutate(ind):
    ind = np.copy(ind)
    i, j = rng.choice(len(ind), size=2, replace=False)
    ind[i], ind[j] = ind[j], ind[i]
    return ind


def ga_step(population, fit):
    #print("DOING A STEP")
    #print("popu to evaluate and generate next", population.shape)
    fitness_population = evaluate_pop(population, fit)
    next_pop, _ = fp_selection(population, fitness_population)
    #print("HERE !", next_pop.shape)
    while len(next_pop) < len(population):
        parent1, _ = tournament_selection(population, fitness_population)
        parent2, _ = tournament_selection(population, fitness_population)
        child = erx.erx(parent1, parent2)
        # child = parent1 + rng.normal(scale=2.5, size=(len(parent1)))
        child = mutate(child)
        next_pop = np.concatenate((next_pop, [child]))
    return next_pop, fitness_population


def ga(x, fit, n_gens=100):
    current_population = x
    fitness_population = evaluate_pop(x, fit)
    best_fitness = np.max(fitness_population)
    x_best = x[np.argmax(fitness_population)]
    for g in range(n_gens):
        print(f"GENERATION {g+1}/{n_gens}")
        new_population, fitness_population = ga_step(current_population, fit)
        max_for_this_generation = np.max(fitness_population)
        if max_for_this_generation > best_fitness:
            best_fitness = max_for_this_generation
            x_best = current_population[np.argmax(fitness_population)]
            print("best fit : ", fit(x_best))
        current_population = new_population
        logging.info('\t%d\t%d', len(x)*g, best_fitness)
        if best_fitness == 0.0:
            print("BEST FITNESS POSSIBLE ACHIEVED")
            break
    return x_best


def fitness(x, s, a, env, params):
    policy = NeuroevoPolicy(s, a)
    policy.set_params(x)
    return evaluate(env, params, policy)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-e', '--env', help='environment', default='small', type=str)
    parser.add_argument('-g', '--gens', help='number of generations', default=100, type=int)
    parser.add_argument('-p', '--pop', help='population size (lambda for the 1+lambda ES)', default=10, type=int)
    parser.add_argument('-s', '--seed', help='seed for evolution', default=0, type=int)
    parser.add_argument('--log', help='log file', default='evolution.log', type=str)
    parser.add_argument('--weights', help='filename to save policy weights', default='weights', type=str)
    args = parser.parse_args()
    logging.basicConfig(filename=args.log, level=logging.DEBUG,
                        format='%(asctime)s %(message)s')

    # starting point
    env, params = get_env(args.env)
    s, a = get_state_action_size(env)
    policy = NeuroevoPolicy(s, a)

    # evolution
    rng = np.random.default_rng(args.seed)
    start = rng.normal(size=(args.pop, len(policy.get_params(),)))

    def fit(x):
        return fitness(x, s, a, env, params)
    x_best = ga(start, fit, args.gens)

    # Evaluation
    policy.set_params(x_best)
    policy.save(args.weights)
    best_eval = evaluate(env, params, policy)
    print('Best individual: ', x_best[:5])
    print('Fitness: ', best_eval)
