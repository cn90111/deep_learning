package metaheuristic;

public class DeParameter
{
	public int size;
	public double f;
	public double solutionLimit;
	public double crossoverRate;

	public DeParameter(int size, double f, double solutionLimit, double initSolutionUpperLimit,
			double initSolutionLowerLimit, double crossoverRate)
	{
		this.size = size;
		this.f = f;
		this.solutionLimit = solutionLimit;
		this.crossoverRate = crossoverRate;
	}
}
