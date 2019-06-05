package metaheuristic;

public class DeSolution
{
	private Solution nowSolution;
	private Solution newSolution;
	private double nowValue = 0;
	private double newValue = 0;
	private double solutionLimit;

	public DeSolution(double solutionLimit)
	{
		this.solutionLimit = solutionLimit;
	}

	public Solution getNowSolution()
	{
		return new Solution(nowSolution);
	}

	public void setNowSolution(Solution nowSolution)
	{
		this.nowSolution = checkLimit(nowSolution, solutionLimit);
	}

	public Solution getNewSolution()
	{
		return new Solution(newSolution);
	}

	public void setNewSolution(Solution newSolution)
	{
		this.newSolution = checkLimit(newSolution, solutionLimit);
	}

	public double getNowValue()
	{
		return nowValue;
	}

	public void setNowValue(double nowSolutionValue)
	{
		this.nowValue = nowSolutionValue;
	}

	public double getNewValue()
	{
		return newValue;
	}

	public void setNewValue(double newSolutionValue)
	{
		this.newValue = newSolutionValue;
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
}
