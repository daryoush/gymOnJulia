
using Flux, Zygote
using OpenAIGym
using Flux.Optimise: update!
using SpecialFunctions: erf


"""
pendulum documentation

Observation
Type: Box(3)

Num	Observation	Min	Max
0	cos(theta)	-1.0	1.0
1	sin(theta)	-1.0	1.0
2	theta dot	-8.0	8.0


The precise equation for reward:

-(theta^2 + 0.1*theta_dt^2 + 0.001*action^2)
Theta is normalized between -pi and pi. Therefore, the lowest reward is -(pi^2 + 0.1*8^2 + 0.001*2^2) = -16.2736044,
and the highest reward is 0. In essence, the goal is to remain at zero angle (vertical), with the least rotational velocity,
and the least effort.


"""

## --

using SpecialFunctions: erf

using Flux
using OpenAIGym
using Zygote
scaledErf(x) =  2 * erf(x)
m = Chain(Dense(3, 50, relu),
              Dense(50, 100, relu),
              Dense(100, 1, scaledErf))

ps=Flux.params(m)

opt=ADAM()


stepErrorFunctionInJulia(state, a) = -(acos(state[1])^2 + 0.1 * state[3]^2 + .001* a[1]^2)

function gameLossFunction(env)
  rs=0

  for i in 1:300
    finished(env) && break
    # do this n times and accumulate the loss
    a= m(state(env))
    stateBeforeStep=state(env)
    Zygote.ignore() do
      r, s = step!(env, state(env),  a )
      @show "after gym step", r, s
    end
    r = stepErrorFunctionInJulia(stateBeforeStep, a)   #doGymStep(env,a)
    Zygote.ignore() do
      @show "my incremental loss", r
      render(env)
    end
    rs+=r
  end
  loss = Flux.mse(rs, 0)
  Zygote.ignore() do
    @show "loss function", loss, rs
    close(env)
  end
  loss
end

function trainPendulum()
  losses=[]
  for i in 1:100
    env = GymEnv("Pendulum-v0")
    reset!(env)
    gs = gradient(ps) do
      l =gameLossFunction(env)
      Zygote.ignore() do
        push!(losses, l)
        @show losses
      end
      l
    end

    Flux.update!(opt, ps, gs)
  end
  losses
end

# for p in ps
#   @show gs[p]
# end
trainPendulum()

## --
