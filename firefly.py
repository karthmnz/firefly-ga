import fire
import torch
import einx

def rbrock(x):
    return (100 * (x[..., 1:] - x[..., :-1] ** 2) ** 2 + (1 - x[..., :-1]) ** 2).sum(dim = -1)

@torch.inference_mode()
def main(
    steps = 5000,
    colonies = 4,
    population_size = 1000,
    dimensions = 12, 
    lower_bound = -4.,
    upper_bound = 4.,
    migrate_every = 50,

    beta0 = 2.,           
    gamma = 1.,  
    alpha = 0.1, 
    alpha_decay = 0.995,


    use_genetic_algorithm = False,
    breed_every = 10,
    tournament_size = 50,
    num_children = 250
):

    assert tournament_size <= population_size
    assert num_children <= population_size

    use_cuda = True
    verbose = True

    cost_function = rbrock

    fireflies = torch.zeros((colonies, population_size, dimensions)).uniform_(lower_bound, upper_bound)

    if torch.cuda.is_available() and use_cuda:
        fireflies = fireflies.cuda()

    device = fireflies.device
    for step in range(steps):


        costs = cost_function(fireflies)

        move_mask = einx.greater('s i, s j -> s i j', costs, costs)

        delta_positions = einx.subtract('s j d, s i d -> s i j d', fireflies, fireflies)

        distance = delta_positions.norm(dim = -1)

        betas = beta0 * (-gamma * distance ** 2).exp()

        attraction = einx.multiply('s i j, s i j d -> s i j d', move_mask * betas, delta_positions)
        random_walk = alpha * (torch.rand_like(fireflies) - 0.5) * (upper_bound - lower_bound)

        fireflies += einx.sum('s i j d -> s i d', attraction) + random_walk

        fireflies.clamp_(min = lower_bound, max = upper_bound)

        alpha *= alpha_decay

        if colonies > 1 and migrate_every > 0 and (step % migrate_every) == 0:
            midpoint = population_size // 2
            fireflies, fireflies_rotate = fireflies[:, :midpoint], fireflies[:, midpoint:]
            migrate_indices = torch.randperm(colonies, device = device)
            fireflies = torch.cat((fireflies, fireflies_rotate[migrate_indices]), dim = 1)

        if not use_genetic_algorithm or (step % breed_every) != 0:
            continue


        cost = cost_function(fireflies)
        fitness = 1. / cost

        batch_randperm = torch.randn((colonies, num_children, population_size), device = device).argsort(dim = -1)
        tournament_indices = batch_randperm[..., :tournament_size]

        participant_fitnesses = einx.get_at('s [p], s c t -> s c t', fitness, tournament_indices)
        winner_tournament_ids = participant_fitnesses.topk(2, dim = -1).indices

        winning_firefly_indices = einx.get_at('s c [t], s c parents -> s c parents', tournament_indices, winner_tournament_ids)

        parent1, parent2 = einx.get_at('s [p] d, s c parents -> parents s c d', fireflies, winning_firefly_indices)

        crossover_mask = torch.rand_like(parent1) < 0.5

        children = torch.where(crossover_mask, parent1, parent2)

        replacement_mask = fitness.argsort(dim = -1).argsort(dim = -1) < num_children

        fireflies[replacement_mask] = einx.rearrange('s p d -> (s p) d', children)


    fireflies = einx.rearrange('s p d -> (s p) d', fireflies)

    costs = cost_function(fireflies)
    sorted_costs, sorted_indices = costs.sort(dim = -1)

    fireflies = fireflies[sorted_indices]

    print(f'best firefly for rbrock with {dimensions} dimensions: {sorted_costs[0]:.3f}: {fireflies[0]}')


if __name__ == '__main__':
    fire.Fire(main)
