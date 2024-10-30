import mesa
import numpy as np
import math
import random as rand

import globals

## Useful snippets:
# specific_agents = model.agents.select(lambda agent:)

class Bee(mesa.Agent):
    def __init__(self, unique_id, model, x, y,
                 A_, wb_, thresh_, trans_prob_,):
                 # scent_thresh, scent_prob):
        super().__init__(unique_id, model)
        self.type = 0

        ## Position vars:
        self.x = x
        self.y = y
        self.old_x = -1
        self.old_y = -1
        self.heading = rand.uniform(0, 2*math.pi)
        self.initial_pos = (x,y)

        ## Food attributes:
        self.hungry = True
        # self.occupied = False
        self.food = 0
        self.counts = 0

        ## Food exchange:
        self.delta_t = 0
        self.times = 0

        ## Neighbor related:
        self.nearby_agents = []
        self.dist_to_neighbors = None

        ## Scenting:
        self.wb = wb_   # 10.0
        self.threshold = thresh_ # * 0.01
        self.A = A_ # * 0.0575
        # grads
        self.grad_x = 0.0
        self.grad_y = 0.0
        self.wx = 0
        self.wy = 0
        self.Cs = []
        self.distances = []     # remove?
        self.total_C = 0
        self.total_grads = np.array([0.0, 0.0])
        # counters
        self.timestep = 0
        self.wait_timestep = 0
        # state and flags
        self.state = 0
        self.next_state = None
        self.prev_state = None
        self.threshold_met = False
        # other:
        self.emission_frequency = 80
        self.sensations = []
        self.trans_prob = trans_prob_ #0.5
        self.wait_period = 20   # original value = 80

        # ## Checks:
        self.zero = False

    # __init__()

    def step(self):
        self.old_x = self.pos[0]
        self.old_y = self.pos[1]

        ## Orient heading if about to scent:
        if self.next_state == 2 and self.state != self.next_state:
            if self.grad_x != 0 and self.grad_y != 0:
                self.set_heading_vector(self.grad_x, self.grad_y)

        ## Do nothing if emitting or waiting, else do one of these
        if self.state == -1 and self.state != self.next_state and self.counts == 0:     # if exiting occupied state
            # print(self.model.t_i, " -- -- ", self.next_state)
            self.state = self.next_state
            self.attempt_move()
        elif self.state != -1 and self.state != 1 and self.state != 2:       # if not emitting or waiting
            ## Update state:
            self.state = self.next_state
            ## Search for nearby agents for trophallaxis:
            self.nearby_agents, self.dist_to_neighbors = self.get_neighbors(1.1)
            if len(self.nearby_agents) > 0:
                ## Pick target to attempt food exchange:
                target = self.nearby_agents[np.argmin(self.dist_to_neighbors)]
                if target:
                    globals.n3_counter += 1
                    # if not target.occupied:
                    if target.state != -1:      # if target is not occupied
                        globals.n2_counter += 1
                        globals.delta_food = (self.food - target.food)
                        if globals.delta_food != 0:
                            globals.n1_counter += 1
                        if globals.delta_food > self.model.troph_thresh:     ####
                            globals.tro_counter += 1
                            # self.occupied = True
                            self.state = -1         # go to occupied state
                            self.delta_t = np.round((globals.delta_food / 2) / globals.food_transfer_rate)
                            self.counts = self.delta_t
                            self.exchange_food(target)
                        else:
                            self.attempt_move()
                    else:
                        self.attempt_move()
                else:
                    self.attempt_move()
            else:
                self.attempt_move()
        else:
            self.state = self.next_state
        if self.hungry:
            self.model.all_fed = False

        if self.state != 2:
            self.__clear()
    # step()

    ########################
    ### Heading Helpers: ###
    ########################

    def normalize_heading(self, head):
        if head < 0:
            return (2*math.pi) + head
        elif head >= 2*math.pi:
            return head - (2*math.pi)
        return head
    # normalize_heading()

    def update_heading(self, min, max):
        tmp = rand.uniform(min, max/2)
        if rand.randint(0,1):
            self.heading = self.heading - tmp
        else:
            self.heading = self.heading + tmp
        self.heading = self.normalize_heading(self.heading)
    # update_heading()

    def set_heading(self, new_head):
        self.heading = self.normalize_heading(new_head)
    # set_heading()

    def set_heading_vector(self, dx, dy):
        h2 = math.atan2(dy, dx)
        return self.normalize_heading(h2)

    ##########################
    ### Movement Function: ###
    ##########################

    ## Only move if random walk or directed walk
    #  - do nothing if emitting or waiting
    def attempt_move(self):
        ## Random Walk:
        if self.state == 0 or self.state == 4:
            ## Update heading:
            self.update_heading(0, self.model.theta)
            ## Move:
            self.forward(1.0)

        ## Directed Walk:
        elif self.state == 3:
            ## Use self.gradient_x and self.gradient_y as heading:
            self.set_heading_vector(self.grad_x, self.grad_y)
            self.update_heading(0, (self.model.theta/2))
            ## Move:
            self.forward(1.0)
    # attempt_move()

    ### forward() + helpers ###

    def _find_next_position(self, step_size, heading):
        next_x = (self.x + (step_size * np.cos(heading)))
        next_y = (self.y + (step_size * np.sin(heading)))
        new_heading = heading
        tmp_x = next_x
        tmp_y = next_y

        ## if out of bounds, then adjust coordinates
        if self.model.space.out_of_bounds((next_x, next_y)):
            tmp1 = False
            tmp2 = False

            ## Check out of bounds X:
            if next_x < 0:
                next_x *= -1
                tmp1 = True
            elif next_x >= self.model.width:
                next_x = self.model.width - (next_x - self.model.width)
                tmp1 = True

            ## Check out of bounds Y:
            if next_y < 0:
                next_y *= -1
                tmp2 = True
            elif next_y >= self.model.height:
                next_y = self.model.height - (next_y - self.model.height)
                tmp2 = True

            ## Fix heading if necessary:
            # if tmp1 or tmp2:
            #     new_heading = self.get_heading_to_point([16,16])
            if tmp1 and tmp2:
                new_heading = rand.uniform(0, 2*math.pi)
            elif tmp1:
                new_heading = (math.pi - heading)
            elif tmp2:
                new_heading = (2*math.pi - heading)

        ## Check edge case:
        if self.model.space.out_of_bounds((next_x, next_y)):
            if next_x == self.model.width:      # if x equal to width
                next_x -= 0.05
            if next_y == self.model.height:     # if y equal to height
                next_x -= 0.05
        return next_x, next_y, self.normalize_heading(new_heading)
    # _find_next_position()

    def _test_small_steps(self, cur_heading):
        ## First try moving forward with smaller step size:
        new_step = 0.1
        tmp_next_x, tmp_next_y, tmp_next_heading = self._find_next_position(new_step, cur_heading)
        able_to_move = self.check_collision((tmp_next_x, tmp_next_y), self.model.bee_positions)
        if able_to_move:
            ## Save values:
            next_x = tmp_next_x
            next_y = tmp_next_y
            next_heading = tmp_next_heading

            ## Increase step size and check collisions:
            for i in range(9):
                new_step += 0.1     # .1  .2  .3  ...  .8  .9  1.0
                new_step = round(new_step, 1)
                tmp_next_x, tmp_next_y, tmp_next_heading = self._find_next_position(new_step, cur_heading)
                if self.check_collision((tmp_next_x, tmp_next_y), self.model.bee_positions):
                    next_x = tmp_next_x
                    next_y = tmp_next_y         # save new position
                    next_heading = tmp_next_heading
                else:
                    break   # use previous new position
        else:
            return -1, -1, -1
        return next_x, next_y, next_heading
    # _test_small_steps()

    def forward(self, step_size):
        if step_size <= 0:
            return

        ## Find new position:
        next_x, next_y, next_heading = self._find_next_position(step_size, self.heading)
        hd_tmp = self.heading

        ## Check for collisions before moving:
        able_to_move = self.check_collision((next_x, next_y), self.model.bee_positions)
        if able_to_move:
            ## Update position internally and in dictionary:
            self.x = next_x
            self.y = next_y
            self.model.bee_positions[f'bee_{self.unique_id}'] = [self.x, self.y]
            self.heading = self.normalize_heading(next_heading)
            ## Change agent's position in Mesa space:
            self.model.space.move_agent(self, (next_x, next_y))
            return
        else:
            ## First try moving forward with smaller step size:
            next_x, next_y, next_heading = self._test_small_steps(self.heading)
            ## If new pos is valid, then move:
            if next_x >= 0:
                self.x = next_x
                self.y = next_y
                self.model.bee_positions[f'bee_{self.unique_id}'] = [self.x, self.y]
                self.heading = self.normalize_heading(next_heading)
                ## Change agent's position in Mesa space:
                self.model.space.move_agent(self, (next_x, next_y))
                return
            ## Next try various step sizes and headings:
            else:
                ## turn 180, then check step sizes:
                new_head = self.heading + rand.uniform(2*math.pi/2, 3*math.pi/2)
                # self.set_heading(self.heading + math.pi)
                next_x, next_y, next_heading = self._test_small_steps(new_head)
                if next_x >= 0:
                    self.x = next_x
                    self.y = next_y
                    self.model.bee_positions[f'bee_{self.unique_id}'] = [self.x, self.y]
                    self.heading = self.normalize_heading(next_heading)
                    ## Change agent's position in Mesa space:
                    self.model.space.move_agent(self, (next_x, next_y))
                    return
                else:
                    self.zero = True
    # forward()

    ###########################
    ### Collision Handling: ###
    ###########################

    def check_collision(self, nextPos, bee_positions):
        ## adapted from Dieu My's trophallaxis model
        distances = []
        for bee_key, bee_xy in bee_positions.items():
            # skip itself:
            if bee_key == f'bee_{self.unique_id}':
                continue
            # Get distance to agent:
            d_i = self.compute_euclidean(nextPos, bee_xy)
            distances.append(d_i)
        distances = np.array(distances)
        threshold_distance = globals.step_size
        able_to_move = np.all(distances > threshold_distance)
        return able_to_move
    # check_collision()

    #################################
    ### Agent Relation Functions: ###
    #################################

    def get_neighbors(self, r):
        nearby_neighbors = []
        dists = []
        my_key = f'bee_{self.unique_id}'
        curPos = self.model.bee_positions[my_key]
        for bee_key, bee_xy in self.model.bee_positions.items():
            ## skip self:
            if bee_key == my_key:
                continue
            ## get dist to agent:
            d_i = self.compute_euclidean(curPos, bee_xy)

            ## If dist < radius, add agent and dist to lists:
            if d_i < r:
                bee_id = int(bee_key[4:])
                nearby_neighbors.append(self.get_agent_id(bee_id))
                dists.append(d_i)
        return nearby_neighbors, dists
    # get_neighbors()

    def get_agent_id(self, bee_id):
        return self.model.agents.select(lambda agent: agent.unique_id == bee_id)[0]
    # get_agent_id()

    ############################
    ### Calculation Helpers: ###
    ############################

    def get_distance_between_points(self, tup1, tup2):
        return self.model.space.get_distance(tup1, tup2)
    # get_distance_between_points()

    def compute_euclidean(self, pos1, pos2):
        dist = np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
        return dist
    # compute_euclidean()

    def get_heading_to_point(self, tup):
        vect_heading = self.model.space.get_heading(self.pos, tup)
        h2 = math.atan2(vect_heading[1], vect_heading[0])
        return self.normalize_heading(h2)
    # get_heading_to_point()

    ######################
    ### Food Exchange: ###
    ######################

    def make_fed(self):
        self.food = 1
        self.hungry = False

    def exchange_food(self, target):
        globals.donor_list.append(self.unique_id)
        globals.foods_list.append(globals.delta_food)
        globals.xcor_list.append(self.pos[0])
        globals.ycor_list.append(self.pos[1])

        ## show target:
        transfer_amount = globals.delta_food / 2
        self.times += 1
        self.food -= transfer_amount
        self.hungry = False

        ## update target's attributes:
        # target.occupied = True
        target.state = -1           # set target to occupied state
        target.hungry = False
        globals.target_list.append(target.unique_id)

        target.food += transfer_amount
        target.counts = self.delta_t
        target.times += 1

        self.update_variance()
    # exchange_food()

    def update_variance(self):
        ## update variance only if food exchange happens:
        globals.var = globals.newvar
        bee_vars = []
        for a in self.model.schedule.agents:
            if a.type == 0:
                bee_vars.append(a.food)
        globals.newvar = np.var(bee_vars)
    # update_variance()

    ###########################
    ### Scenting Functions: ###
    ###########################

    ######################
    ### Step Functions ###

    def sense_environment(self, env, pheromone_src, pheromone_src_C):
        ## Look at scents and determine effect
        # Calculate gradient at agent's position:
        grad = env.calc_gradient_to_source(self.model.t_i, self.x, self.y, pheromone_src)

        ## Calc concentration at bee.x, bee.y
        x_bee = self.model.environment.convert_xy_to_index(self.x)
        y_bee = self.model.environment.convert_xy_to_index(self.y)
        concentration_at_bee = pheromone_src_C[int(y_bee), int(x_bee)]

        ## Omit very low sources, threshold at lowest T in search:
        if concentration_at_bee > 1e-3:
            ## Calc distance between bee
            self_bee_position = np.array([x_bee, y_bee])

            x_bee_src = self.model.environment.convert_xy_to_index(pheromone_src['x'])
            y_bee_src = self.model.environment.convert_xy_to_index(pheromone_src['y'])

            src_bee_position = np.array([x_bee_src, y_bee_src])

            distance_between_bees = np.linalg.norm(src_bee_position-self_bee_position)

            ## Update bee's sensations:
            sensation = {
                "bee_id": pheromone_src['bee_id'],
                "C": concentration_at_bee,
                "grad": grad,
                "distance": distance_between_bees
            }
            self.sensations.append(sensation)
    # sense_environment()

    ##########################
    #### Update Functions ####
    ##########################

    def fed_update(self):
        self.prev_state = self.state
        self.next_state = None

        ## Occupied Case:
        if self.state == -1:
            if self.counts > 0:
                self.counts -= 1
            else:
                self.next_state = 0

        ## Wait case:
        elif self.state == 2:
            if self.timestep < self.emission_frequency:
                self.timestep += 1
            else:
                self.state = 0      # go to random walk

        ## If currently scenting case:
        elif self.state == 4:
            if self.timestep < self.emission_frequency*0.7:
                self.timestep += 1     # increment counter
            else:
                if self.model.fed_scent_move:
                    self.next_state = 0     # next state = random walk
                else:
                    self.next_state = 2     # next state = wait

        ## Random or directed walk case:
        else:
            random_draw = np.random.uniform(0,1)    # random chance to start scenting
            if random_draw <= self.model.fed_scent_prob:
                self.next_state = 4         # start scent walking
                self.timestep = 0
            else:
                self.next_state = 0         # random walk

        ## Check if state hasn't changed:
        if self.next_state == None:
            self.next_state = self.state

        ## Remove scenting velocity (want even spread):
        self.wx = 0
        self.wy = 0
        ## And remove gradient heading:
        self.grad_x = 0
        self.grad_y = 0

        # ## Update state:
        # self.state = self.next_state
    # fed_update()

    def update(self):
        # if self.sensitivity_mode == 'none':
        self.__determine_sensation_effects_mode_1()
        # elif self.sensitivity_mode == 'queen_worker':
        #     self.__determine_sensation_effects_mode_2()
        # elif self.sensitivity_mode == 'all':
        #     self.__determine_sensation_effects_mode_3()

        ## Check if threshold met:
        if self.total_C > self.threshold:
            self.threshold_met = True
            # Don't compute gradient and bias when emitting
            # Compute when emitting is over
            if self.state != 1 and self.state != -1:
                # print(" -2")
                self.__update_gradient(self.total_grads)
                self.__normalize_gradient()
                self.__update_bias()
                # self.__check_src_contributions()        # Not really used for anything
        else:
            self.threshold_met = False

        ## Update state:
        self.__update_state()
    # update()

    #### Update Helpers ####

    def __determine_sensation_effects_mode_1(self):
        """
            # 1. No distinction between any bees (e.g., pheromones from queen are treated equal to workers)
        """
        self.total_C = 0
        self.total_grads = np.array([0.0, 0.0])
        for sensation in self.sensations:
            self.total_C += sensation['C']
            self.total_grads += np.array(sensation['grad'])
    # __determine_sensation_effects_mode_1()

    def __update_gradient(self, grad):
        grad_x, grad_y = grad
        self.grad_x += grad_x
        self.grad_y += grad_y
    # __update_gradient()

    def __normalize_gradient(self):
        d = np.linalg.norm([self.grad_x, self.grad_y])
        self.grad_x = self.grad_x / (d + 1e-9)
        self.grad_y = self.grad_y / (d + 1e-9)
    # __normalize_gradient()

    def __update_bias(self):
        self.wx = -self.grad_x
        self.wy = -self.grad_y
    # __update_bias()

    def __check_src_contributions(self):
        self.Cs = []
        self.distances = []
        for sensation in self.sensations:
            self.Cs.append(sensation['C'])
            self.distances.append(sensation['distance'])
    # __check_src_contributions()

    ##########################
    ###### Update State ######
    ##########################

    ## Possible States:
    #  -1: Occupied
    #   0: random_walk
    #   1: emit             (2)
    #   2: wait             (3)
    #   3: directed_walk    (4)
    #   4: emit and random walking

    def __update_state(self):
        self.prev_state = self.state
        self.next_state = None

        ## Occupied with Trophallaxis Case:
        if self.state == -1:
            if self.counts > 0:     # increment counter
                self.counts -= 1
            else:
                self.next_state = 0     # go to random walk if done

        ## Random Walk Pre/Post Case:
        elif self.threshold_met and self.state == 0: # or self.state == 1):
            random_draw = np.random.uniform(0,1)
            if random_draw <= self.trans_prob:
                self.next_state = 1         # emit
                self.wait_timestep = 0
            else:
                self.next_state = 3         # directed walk

        ## Emit Case:
        ## scent for half (or whatever fraction) of wait_period, and
        ##  for the rest wait in place for pheromone to decay..
        elif self.state == 1:
            if self.wait_period*0.5 <= self.wait_timestep <= self.wait_period:
                self.next_state = 2         # wait
            self.wait_timestep += 1

        ################## Only Wait for 1 Timestep ###########
        ## Wait Case:  Agent stands still
        ##  so we don't need emit_final state to resume calculating gradient
        elif self.state == 2:
            if self.wait_timestep > 0: # self.wait_period:
                self.next_state = 3         # directed walk
            self.wait_timestep += 1
        #######################################################

        ## Directed Walk case:
        elif self.state == 3:
            if self.threshold_met:
                random_draw = np.random.uniform(0,1)
                if random_draw <= self.trans_prob:
                    self.next_state = 1         # emit
                    self.wait_timestep = 0
            else:
                self.next_state = 0 #1     # random walk post

        ## Inactive Case
        elif self.state == -1:
            self.next_state = -1

        ## Check if state hasn't changed:
        if self.next_state is None:
            self.next_state = self.state

        # ## Update state:
        # self.state = self.next_state
    # __update_state()

    def __clear(self):
        self.grad_x = 0
        self.grad_y = 0
        self.sensations = []
    # clear()

# Class Bee()
