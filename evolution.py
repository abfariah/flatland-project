from evaluate import get_env, get_state_action_size, evaluate
from policy import NeuroevoPolicy
from argparse import ArgumentParser
import numpy as np
import logging
from tqdm import trange, tqdm

def random_search(x, fitness, gens, std=0.01, r=5., rng=np.random.default_rng()):
    x_best = x
    for g in trange(gens):
        x_temp = np.random.uniform(-r, r, len(x_best))
        if fitness(x_temp) > fitness(x_best):
            x_best = x_temp
    return x_best

def random_optimization(x, fitness, gens, std=0.01, r=5., rng=np.random.default_rng()):
    x_best = x
    for g in trange(gens):
        N = rng.normal(size=(len(x))) * std
        x_temp = x + N[:]
        if fitness(x_temp) > fitness(x_best):
            x_best = x_temp
    return x_best

def oneplus_lambda(x, fitness, gens, lam, std=0.5, rng=np.random.default_rng()):
    x_best = x
    f_best = -np.Inf
    n_evals = 0
    for g in trange(gens):
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



def mu_lambda(x, fitness, gens, lam, alpha=0.2, verbose=False):
    x_best = x
    f_best = -np.Inf
    n_evals = 0
    fits = np.zeros(gens)
    for g in range(gens):
        N = np.random.normal(size=(lam, len(x))) *2
        F = np.zeros(lam)
        for i in range(lam):
            ind = x + N[i, :]
            F[i] = fitness(ind)
            print("F[" + str(i)+ "] =" + str(F[i]))
            if F[i] > f_best:
                f_best = F[i]
                x_best = ind
        fits[g] = f_best
        mu_f = np.mean(F)
        std_f = np.std(F)
        A = F
        if std_f != 0:
            A = (F - mu_f) / std_f
        x = x - alpha * np.dot(A, N) / lam
        n_evals += lam
        logging.info('\t%d\t%d', n_evals, f_best)
        print("x0 = "+str(x[0]))
    return x_best

#def mu_plus_rho_lambda():
#    Initialize x randomly in ℝ
#    while not terminate
#        for i in [1,λ]
#            N_i = 𝑁(0, 1)
#            F_i = f(x + N_i)
#        A = (F−𝜇(F))/𝜎(F)
#        x = x - 𝛼(A⋅N)/𝜆


def simulated_annealing_proba(f, f_best, t):
    return np.exp(-(f - f_best) / t)

def simulated_annealing_optimization(x, fitness, gens, std=0.01, rng=np.random.default_rng()):
    x_best = x
    f_best = -np.Inf
    n_evals = 0
    for k in trange(gens):
        N = rng.normal(size=(len(x))) * std
        ind = x + N[:]
        f = fitness(ind)
        if f > f_best or (rng.random() < simulated_annealing_proba(f, f_best, k)):
            f_best = f
            x_best = ind
        n_evals += 1
        logging.info('\t%d\t%d', n_evals, f_best)
    return x_best





#def truncation_selection(population, fitness, p=0.2):
#    n_elites = int(np.floor(len(population) * p))
#    elites = np.argsort(fitness)[:n_elites]
#    return population[elites], fitness[elites]
#
#def fp_selection(population, fitness):
#    p = (np.max(fitness) - fitness)
#    p /= np.sum(p)
#    rng = np.random.default_rng()
#    ind = rng.choice(len(population), p=p)
#    return population[ind], fitness[ind]
#
#def tournament_selection(population, fitness, t_size=3):
#    inds = rng.choice(len(population), t_size)
#    ind = inds[np.argmin(fitness[inds])]
#    return population[ind], fitness[ind]
#
#def one_point(p1, p2):
#    rng = np.random.default_rng()
#    x = rng.choice(np.arange(1, np.minimum(len(p1)-1, len(p2)-1)))
#    return np.concatenate((p1[:x], p2[x:])), np.concatenate((p2[:x],p1[x:]))
#
#def mutate(ind):
#    ind = np.copy(ind)
#    rng = np.random.default_rng()
#    i, j = rng.choice(len(ind), size=2, replace=False)
#    ind[i], ind[j] = ind[j], ind[i]
#    return ind
#
#def ga_step(population):
#    fitness = evaluate(population, d)
#    next_pop, _ = truncation_selection(population, fitness)
#    while len(next_pop) < len(population):
#        parent1, _ = tournament_selection(population, fitness)
#        parent2, _ = tournament_selection(population, fitness)
#        child = erx.erx(parent1, parent2)
#        child = mutate(child)
#        next_pop = np.concatenate((next_pop, [child]))
#    return next_pop, fitness
#
#def ga(n_population=100, n_gen=100, verbose=False):
#    population = np.array([rng.permutation(n_cities) for i in range(n_population)])
#    minfit = np.zeros(n_gen)
#    t = trange(n_gen)
#    for i in t:
#        population, fitness = ga_step(population)
#        minfit[i] = np.min(fitness)
#        if i > 2 and minfit[i] < minfit[i-1]:
#            t.set_description(f"{i: 03d}, {minfit[i]: 5.3f}")


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
    parser.add_argument('-S', '--std', help='the standard deviation of the search', default=0.01, type=float)
    parser.add_argument('-r', '--range', help='the range of the search', default=5., type=float)
    parser.add_argument('-a', '--algorithm', help='the algorithm', default="opl", type=str)
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
    start = rng.normal(size=(len(policy.get_params(),)))

    def fit(x):
        return fitness(x, s, a, env, params)

    print(args)
    if args.algorithm == "opl":
        x_best = oneplus_lambda(start, fit, args.gens, args.pop, rng=rng)
    elif args.algorithm == "rs":
        x_best = random_search(start, fit, args.gens, std=args.std, r=args.range, rng=np.random.default_rng())
    elif args.algorithm == "ro":
        x_best = random_optimization(start, fit, args.gens, std=args.std, r=args.range, rng=np.random.default_rng())
    elif args.algorithm == "sao":
        x_best = simulated_annealing_optimization(start, fit, args.gens, std=args.std, rng=np.random.default_rng())
    elif args.algorithm == "mu":
        x_best =mu_lambda(start, fit, args.gens, args.pop)
    else:
        print(f"unkown algorithm '{args.algorithm}'. Aborting.")
        exit()


    # Evaluation
    policy.set_params(x_best)
    policy.save(args.weights)
    best_eval = evaluate(env, params, policy)
    print('Best individual: ', x_best[:5])
    print('Fitness: ', best_eval)
