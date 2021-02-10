using OpenAIGym
using Reinforce
using Images
using Plots
using PyPlot
## --
Reinforce.action(policy::AbstractPolicy, r, s, A) = rand(A)

function basic()

  env = GymEnv("Pong-v0")   #(:CartPole, :v0)
  reward = 0
  episode_count = 5

  try
    for i=1:episode_count
        total = 0
        ob = reset!(env)
        render(env)#comment out this line if you do not want to visualize the environment
        for i ∈ 1:50
            action = rand(env.actions)
            reward, s = step!(env, action)
            @show total += reward
            render(env)#comment out this line if you do not want to visualize the environment
            #done && break
        end
        println("episode $i total Rewards: $total")
    end
  finally
    close(env)
  end
end


basic()

## --



function allRandom()
  env = GymEnv(:CartPole, :v0)
  for i ∈ 1:20
    T = 0
    R = run_episode(env, RandomPolicy()) do (s, a, r, s′)
      render(env)
      T += 1
    end
    @info("Episode $i finished after $T steps. Total reward: $R")
  end
  close(env)
end

allRandom()

## --

## Manually running the game without episode

env = GymEnv("Pong-v0")
render(env)
initState=reset!(env)
cropState=initState[35:194,:,1] ## convert to 160x160 picture of only the first layer of image
heatmap(cropState)    ## to see the image

s = state(env)
A = actions(env, s)  # action space
r = reward(env)
a=0
snew=s
r, snew = step!(env, snew, a)

render(env)

for i in 1:1000
  a=rand(A.items)  ## pick a random action
  r, snew = step!(env, snew, a)
  render(env)
end
