const N_ACTIONS = 7
const ACTIONS_DICT = Dict(:left   => 1,
                          :right  => 2,
                          :up     => 3,
                          :down   => 4,
                          :scoop  => 5,
                          :accept => 6,
                          :reject => 7)
                
const ACTION_DIRS = (pos(-1,0),
                    pos(0,),
                    pos(1,0),
                    pos(0,-1),
                    pos(-1,0))