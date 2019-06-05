package metaheuristic;

public class DeParameter
{
	public int size;
	public double f;
	public double solutionLimit;
	public double initSolutionUpperLimit;
	public double initSolutionLowerLimit;
	public double crossoverRate;

	// DE/best/1/bin strategy
	public int updateMode;// best or random
	public int groupNumber;// 1,2,3....

	public int totalReferenceCount;

	public DeParameter(int size, double f, double solutionLimit, double initSolutionUpperLimit,
			double initSolutionLowerLimit, double crossoverRate, int updateMode, int groupNumber)
	{
		this.size = size;
		this.f = f;
		this.solutionLimit = solutionLimit;
		this.initSolutionUpperLimit = initSolutionUpperLimit;
		this.initSolutionLowerLimit = initSolutionLowerLimit;
		this.crossoverRate = crossoverRate;
		this.updateMode = updateMode;
		this.groupNumber = groupNumber;
		this.totalReferenceCount = 1 + 2 * groupNumber;
	}
}
