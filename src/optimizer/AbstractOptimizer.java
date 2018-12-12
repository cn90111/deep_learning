package optimizer;

public abstract class AbstractOptimizer
{
	public AbstractOptimizer()
	{
	}

	public abstract double[][][] updateWeight(double error);
}
