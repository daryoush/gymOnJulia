using Flux
using OpenAIGym
import  Reinforce: action

function stateToInput(s)
  cropState=s[35:194,:,1] ## convert to 160x160 picture of only the first layer of image

  ### TODO: maintain a history, be able to reset the history, figure out how to just send a diff
  ## as input
end


env = GymEnv("Pong-v0")
render(env)
initState=reset!(env)

s = state(env)
A = actions(env, s)  # action space
r = reward(env)

try
  for i in 1:1000
    a=rand(A.items)  ## pick a random action
    r, s = step!(env, s, a)
    render(env)
  end

finally
  close(env)
end
