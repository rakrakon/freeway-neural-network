classDiagram
    direction LR

    class MainScript {
        +main()
        +draw_text(surface, text, font, color, x, y, center)
        +show_main_menu(screen)
        +show_game_over(screen, info)
        +play_manual(env, graphics)
        +play_ai(env, graphics, nn_player)
    }

    class RunTrainingScript {
        +main()
    }

    class Graphics {
        +config : Config
        +screen_width : int
        +screen_height : int
        +num_lanes : int
        +screen
        +car_sprite
        +chicken_sprite
        +font
        +clock
        +render(obs, score)
        -_draw_environment(scale)
        -_process_obs(obs)
    }

    class FreewayENV {
        +config : Config
        +height : int
        +width : int
        +action_space : ActionSpace
        +player_x : int
        +player_y : int
        +prev_player_y : int
        +score : int
        +cars : list[Car]
        +frame : int
        +max_frames : int
        +skip_rate : int
        +reset()
        +step(action)
        -_init_cars()
        -_reset_player()
        -_move_cars()
        -_draw_cars(frame)
        -_draw_player(frame)
        -_get_obs()
        -_check_collision() bool
    }

    class Car {
        +x : float
        +y : float
        +lane : int
        +speed : int
        +direction : int
    }

    class ActionSpace {
        +n : int
        +sample() int
        +seed(seed)
    }

    class DQNTrainer {
        +logger
        +env : FreewayENV
        +train_config : Config
        +device
        +frame_buffer : deque
        +steps_done : int
        +policy_net : DuelingDQN
        +target_net : DuelingDQN
        +optimizer
        +memory : ReplayMemory
        +start_episode : int
        +get_state(observation)
        +reset_frame_buffer()
        +select_action(state)
        +get_epsilon()
        +optimize_model()
        +train()
        +plot_training_progress(rewards, lengths, losses, show_result)
        +plot_durations(show_result)
    }

    class DQNPlayer {
        +logger
        +train_config : Config
        +device
        +policy_net : DuelingDQN
        +frame_buffer : deque
        +reset(initial_observation)
        +get_action(observation) int
    }

    class DuelingDQN {
        +conv1
        +conv2
        +conv3
        +value_fc1
        +value_fc2
        +advantage_fc1
        +advantage_fc2
        -_initialize_weights()
        +forward(x)
    }

    class ReplayMemory {
        +memory : deque
        +push(*args)
        +sample(batch_size)
        +__len__() int
    }

    class Config {
        +width : int
        +height : int
        +num_lanes : int
        +frame_stack : int
        +frame_height : int
        +frame_width : int
        +learning_rate : float
        +gamma : float
        +epsilon_start : float
        +epsilon_end : float
        +epsilon_decay : int
        +batch_size : int
        +replay_buffer_size : int
        +target_update_frequency : int
        +learning_starts : int
        +graphics_width : int
        +graphics_height : int
    }

    MainScript ..> FreewayENV : composition
    MainScript ..> Graphics : composition
    MainScript ..> DQNPlayer : composition

    RunTrainingScript ..> DQNTrainer : composition

    Graphics ..> Config : composition

    FreewayENV *-- ActionSpace : composition
    FreewayENV *-- Car : composition
    FreewayENV ..> Config : composition

    DQNTrainer *-- FreewayENV : composition
    DQNTrainer *-- ReplayMemory : composition
    DQNTrainer *-- DuelingDQN : composition
    DQNTrainer *-- DuelingDQN : composition
    DQNTrainer ..> Config : composition

    DQNPlayer *-- DuelingDQN : composition
    DQNPlayer ..> Config : composition