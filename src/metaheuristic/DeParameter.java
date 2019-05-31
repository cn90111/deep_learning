package metaheuristic;

import optimizer.DifferentialEvolutionBackPropagation;

public class DeParameter
{
	public int size;
	public double f;
	public double solutionLimit;
	public double crossoverRate;

	// DE/best/1/bin strategy
	public int updateMode;// best or random
	public int groupNumber;// 1,2,3....

	public int totalRandom;

	public DeParameter(int size, double f, double solutionLimit, double initSolutionUpperLimit,
			double initSolutionLowerLimit, double crossoverRate, int updateMode, int groupNumber)
	{
		this.size = size;
		this.f = f;
		this.solutionLimit = solutionLimit;
		this.crossoverRate = crossoverRate;
		this.updateMode = updateMode;
		this.groupNumber = groupNumber;

		switch (updateMode)
		{
			case DifferentialEvolutionBackPropagation.UPDATE_MODE_BEST:
				totalRandom = 0;
				break;
			case DifferentialEvolutionBackPropagation.UPDATE_MODE_RANDOM:
				totalRandom = 1;
				break;
			default:
				throw new UnsupportedOperationException("Only support Best and Random.");
		}
		totalRandom = totalRandom + 2 * groupNumber;
	}
}
