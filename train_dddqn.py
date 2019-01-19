import tensorflow as tf
import numpy as np
import random
from collections import deque
from Default.network_dddqn import DDDQNNet
from mlagents.envs import UnityEnvironment

state_size = 195 * 3
action_size = 6

learning_rate =  0.00025
total_episodes = 100000         # Total episodes for training
max_steps = 300              # Max possible steps in an episode
batch_size = 128

explore_start = 1.0            # exploration probability at start
explore_stop = 0.1            # minimum exploration probability
decay_rate = 0.00005            # exponential decay rate for exploration prob
gamma = 0.95

pretrain_length = 100  # Number of experiences stored in the Memory when initialized for the first time
memory_size = 100000       # Number of experiences the Memory can keep

train_mode = False
load_model = False
epsilon_refresh = True

run_episode = 100000
test_episode = 1000

class dddqnTrain():

    def __init__(self, model_name):
        self.model_name = model_name
        self.DQNetwork = DDDQNNet(name= model_name + "DQNetwork")
        # Instantiate the target network
        self.TargetNetwork = DDDQNNet(name= model_name + "TargetNetwork")

    def act(self, states, sess):
        states = np.reshape(states, newshape=(-1, state_size))
        if train_mode == True and epsilon > np.random.rand():
            Qs = sess.run(self.DQNetwork.output, feed_dict={self.DQNetwork.inputs_: states})
            action = np.argmax(Qs)
            return action
        else:
            Qs = sess.run(self.DQNetwork.output, feed_dict={self.DQNetwork.inputs_: states})
            action = np.argmax(Qs)
            return action

    def update_target_graph(self, sess):
        from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.model_name + "DQNetwork")
        to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.model_name + "TargetNetwork")
        op_holder = []
        for from_var, to_var in zip(from_vars, to_vars):
            op_holder.append(to_var.assign(from_var))
        sess.run(op_holder)

def append_sample(memory, state, action, reward, next_state, done):
    memory.append((state, action, reward, next_state, done))

if __name__ == "__main__":

    # DQN Train 시작 ====================================================================================================
    #
    env_name =
    save_path =
    load_path = 

    agent1 = dddqnTrain("agent1")
    agent2 = dddqnTrain("agent2")

    memory1 = deque(maxlen=memory_size)
    memory2 = deque(maxlen=memory_size)

    env = UnityEnvironment(file_name=env_name, worker_id=1)

    default_brain = env.brain_names[0]
    brain = env.brains[default_brain]
    brain_name1 = env.brain_names[0]
    brain_name2 = env.brain_names[1]
    brain1 = env.brains[brain_name1]
    brain2 = env.brains[brain_name2]


    # Save & Load ============================================
    Saver = tf.train.Saver(max_to_keep=5)
    load_path = load_path
    # self.Summary,self.Merge = self.make_Summary()
    # ========================================================

    # Session Initialize =====================================
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.15
    sess = tf.Session(config=config)

    if load_model == True:
        ckpt = tf.train.get_checkpoint_state(load_path)
        Saver.restore(sess, ckpt.model_checkpoint_path)
        print("[Restore Model]")
    else:
        init = tf.global_variables_initializer()
        sess.run(init)
        print("[Initialize Model]")

    # Reset Environment =======================
    env_info = env.reset(train_mode=train_mode)
    print("[Env Reset]")
    # =========================================

    # Update the parameters of our TargetNetwork with DQN_weights
    agent1.update_target_graph(sess)
    agent2.update_target_graph(sess)

    step = 0
    tau1 = 0
    tau2 = 0

    for episode in range(run_episode + test_episode):
        if episode > run_episode:
            train_mode = False

        losses1 = []
        losses2 = []
        episode_rewards1 = []
        episode_rewards2 = []

        env_info = env.reset(train_mode=train_mode)
        state1 = env_info[brain_name1].vector_observations[0]
        done1 = False
        state2 = env_info[brain_name2].vector_observations[0]
        done2 = False
        done = done1 or done2

        while not done:
            step += 1

            action1 = agent1.act(state1, sess)
            action2 = agent2.act(state2, sess)

            env_info = env.step(vector_action={brain_name1: [action1], brain_name2: [action2]})

            next_state1 = env_info[brain_name1].vector_observations[0]
            reward1 = env_info[brain_name1].rewards[0]
            done1 = env_info[brain_name1].local_done[0]
            episode_rewards1.append(reward1)

            next_state2 = env_info[brain_name2].vector_observations[0]
            reward2 = env_info[brain_name2].rewards[0]
            done2 = env_info[brain_name2].local_done[0]
            episode_rewards2.append(reward2)

            ac1 = np.zeros(shape=[action_size])
            ac1[action1] = 1
            action1 = ac1

            ac2 = np.zeros(shape=[action_size])
            ac2[action2] = 1
            action2 = ac2

            done = done1 or done2

            # If the game is finished
            if done:

                # Get the total reward of the episode
                total_reward1 = np.sum(episode_rewards1)
                total_reward2 = np.sum(episode_rewards2)

                print('Step {} /Episode: {} /'.format(step, episode),
                      'Reward 1 : {} /'.format(total_reward1),
                      'Reward 2 : {} /'.format(total_reward2),
                      'Loss 1 : {:.4f} /'.format(np.mean(losses1)),
                      'Loss 2 : {:.4f} /'.format(np.mean(losses2)))

                # Add experience to memory
                experience1 = state1, action1, reward1, next_state1, done1
                append_sample(memory1, state1, action1, reward1, next_state1, done1)
                experience2 = state2, action2, reward2, next_state2, done2
                append_sample(memory2, state2, action2, reward2, next_state2, done2)

            else:
                experience1 = state1, action1, reward1, next_state1, done1
                append_sample(memory1, state1, action1, reward1, next_state1, done1)
                experience2 = state2, action2, reward2, next_state2, done2
                append_sample(memory2, state2, action2, reward2, next_state2, done2)

                # st+1 is now our current state
                state1 = next_state1
                state2 = next_state2

            if len(memory1) > batch_size and len(memory2) > batch_size:

                mini_batch1 = random.sample(memory1, batch_size)

                states1 = []
                actions1 = []
                rewards1 = []
                next_states1 = []
                dones1 = []

                for i in range(batch_size):
                    states1.append(mini_batch1[i][0])
                    actions1.append(mini_batch1[i][1])
                    rewards1.append(mini_batch1[i][2])
                    next_states1.append(mini_batch1[i][3])
                    dones1.append(mini_batch1[i][4])

                states_mb1 = np.reshape(states1, newshape=(-1, state_size))
                actions_mb1 = np.reshape(actions1, newshape=(-1, action_size))
                rewards_mb1 = np.reshape(rewards1, newshape=(-1,))
                next_states_mb1 = np.reshape(next_states1, newshape=(-1, state_size))
                dones_mb1 = np.reshape(dones1, newshape=(-1,))

                mini_batch2 = random.sample(memory2, batch_size)
                states2 = []
                actions2 = []
                rewards2 = []
                next_states2 = []
                dones2= []

                for i in range(batch_size):
                    states2.append(mini_batch2[i][0])
                    actions2.append(mini_batch2[i][1])
                    rewards2.append(mini_batch2[i][2])
                    next_states2.append(mini_batch2[i][3])
                    dones2.append(mini_batch2[i][4])

                states_mb2 = np.reshape(states2, newshape=(-1, state_size))
                actions_mb2 = np.reshape(actions2, newshape=(-1, action_size))
                rewards_mb2 = np.reshape(rewards2, newshape=(-1,))
                next_states_mb2 = np.reshape(next_states2, newshape=(-1, state_size))
                dones_mb2 = np.reshape(dones2, newshape=(-1,))

                target_Qs_batch1 = []
                target_Qs_batch2 = []

                ### DOUBLE DQN Logic
                # Use DQNNetwork to select the action to take at next_state (a') (action with the highest Q-value)
                # Use TargetNetwork to calculate the Q_val of Q(s',a')

                # For Doubling DQN
                q_next_state1 = sess.run(agent1.DQNetwork.output, feed_dict={agent1.DQNetwork.inputs_: next_states_mb1})
                q_next_state2 = sess.run(agent2.DQNetwork.output, feed_dict={agent2.DQNetwork.inputs_: next_states_mb2})
                q_next_state1 = np.reshape(q_next_state1, newshape=(-1, action_size))
                q_next_state2 = np.reshape(q_next_state2, newshape=(-1, action_size))

                # Calculate Qtarget for all actions that state
                q_target_next_state1 = sess.run(agent1.TargetNetwork.output, feed_dict={agent1.TargetNetwork.inputs_: next_states_mb1})
                q_target_next_state2 = sess.run(agent2.TargetNetwork.output, feed_dict={agent2.TargetNetwork.inputs_: next_states_mb2})
                q_target_next_state1 = np.reshape(q_target_next_state1, newshape=(-1, action_size))
                q_target_next_state2 = np.reshape(q_target_next_state2, newshape=(-1, action_size))

                for i in range(0, batch_size):
                    terminal1 = dones_mb1[i]
                    action1 = np.argmax(q_next_state1[i])
                    if terminal1:
                        target_Qs_batch1.append(rewards_mb1[i])
                    else:
                        target1 = rewards_mb1[i] + gamma * q_target_next_state1[i][action1]
                        target_Qs_batch1.append(target1)

                targets_mb1 = np.reshape(target_Qs_batch1, newshape=(-1,))
                _, loss1= sess.run(
                    [agent1.DQNetwork.optimizer, agent1.DQNetwork.loss],
                    feed_dict={agent1.DQNetwork.inputs_: states_mb1,
                               agent1.DQNetwork.target_Q: targets_mb1,
                               agent1.DQNetwork.actions_: actions_mb1})
                losses1.append(loss1)


                for i in range(0, batch_size):
                    terminal2 = dones_mb2[i]
                    action2 = np.argmax(q_next_state2[i])
                    if terminal2:
                        target_Qs_batch2.append(rewards_mb2[i])
                    else:
                        target2 = rewards_mb2[i] + gamma * q_target_next_state2[i][action2]
                        target_Qs_batch2.append(target2)

                targets_mb2 = np.reshape(target_Qs_batch2, newshape=(-1,))
                _, loss2 = sess.run(
                    [agent2.DQNetwork.optimizer, agent2.DQNetwork.loss],
                    feed_dict={agent2.DQNetwork.inputs_: states_mb2,
                               agent2.DQNetwork.target_Q: targets_mb2,
                               agent2.DQNetwork.actions_: actions_mb2})
                losses2.append(loss2)

                if step % 1000 == 0:
                    agent1.update_target_graph(sess)
                    agent2.update_target_graph(sess)


        if episode % 100 == 0 and episode != 0:
            Saver.save(sess, save_path + "/model.ckpt")
            print("Save Model {}".format(episode))
