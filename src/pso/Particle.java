package pso;

public class Particle
{
	private double nowValue = 0;
	private double localBestValue = 0;
	private Solution nowSolution;
	private Solution localBestSolution;
	private Solution velocity;
	private double velocityLimit;
	private double solutionLimit;

	public Particle(double velocityLimit, double solutionLimit)
	{
		this.velocityLimit = velocityLimit;
		this.solutionLimit = solutionLimit;
	}

	public void setNowValue(double value)
	{
		nowValue = value;
	}

	public double getNowValue()
	{
		return nowValue;
	}
	
	public void setLocalBestValue(double value)
	{
		localBestValue = value;
	}

	public double getLocalBestValue()
	{
		return localBestValue;
	}

	public void setLocalBestSolution(Solution solution)
	{
		localBestSolution = solution;
	}

	public Solution getLocalBestSolution()
	{
		return new Solution(localBestSolution);
	}

	public void setVelocity(Solution velocity)
	{
		this.velocity = checkLimit(velocity, velocityLimit);
	}

	public Solution getVelocity()
	{
		return new Solution(velocity);
	}

	public void setNowSolution(Solution solution)
	{
		nowSolution = checkLimit(solution, solutionLimit);
	}

	private Solution checkLimit(Solution solution, double limit)
	{
		double[][][] weight = solution.getWeight();
		double[][] bias = solution.getBias();
		for (int i = 0; i < weight.length; i++)
		{
			for (int j = 0; j < weight[i].length; j++)
			{
				for (int k = 0; k < weight[i][j].length; k++)
				{
					if (weight[i][j][k] > limit)
					{
						weight[i][j][k] = limit;
					}
					else if (weight[i][j][k] < -1 * limit)
					{
						weight[i][j][k] = -1 * limit;
					}
				}

				if (bias[i][j] > limit)
				{
					bias[i][j] = limit;
				}
				else if (bias[i][j] < -1 * limit)
				{
					bias[i][j] = -1 * limit;
				}
			}
		}
		solution.setWeight(weight);
		solution.setBias(bias);
		return solution;
	}

	public Solution getNowSolution()
	{
		return new Solution(nowSolution);
	}
}
