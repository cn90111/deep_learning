package metaheuristic;

public class PsoParameter
{
	public int size;
	public double c1;
	public double c2;
	public double maxW;
	public double linearEndW;
	public double linearEndCount;
	public double nonlinearlyWeight;

	public double velocityLimit;
	public double initVelocityUpperLimit;
	public double initVelocityLowerLimit;

	public double solutionLimit;
	public double initSolutionUpperLimit;
	public double initSolutionLowerLimit;

	public PsoParameter(int size, double globalSearchWeight, double localSearchWeight, double velocityRate,
			double velocityLimit, double initVelocityUpperLimit, double initVelocitylowerLimit, double solutionLimit,
			double initSolutionUpperLimit, double initSolutionlowerLimit, double linearEndCount,
			double nonlinearlyWeight)
	{
		this.size = size;
		this.c1 = localSearchWeight;
		this.c2 = globalSearchWeight;
		this.maxW = velocityRate;
		this.linearEndW = maxW / 2;
		this.velocityLimit = velocityLimit;
		this.linearEndCount = linearEndCount;
		this.nonlinearlyWeight = nonlinearlyWeight;

		this.velocityLimit = velocityLimit;
		this.initVelocityUpperLimit = initVelocityUpperLimit;
		this.initVelocityLowerLimit = initVelocitylowerLimit;

		this.solutionLimit = solutionLimit;
		this.initSolutionUpperLimit = initSolutionUpperLimit;
		this.initSolutionLowerLimit = initSolutionlowerLimit;
	}
}
