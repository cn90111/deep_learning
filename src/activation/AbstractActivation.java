package activation;

public abstract class AbstractActivation
{
	public AbstractActivation()
	{
	}

	public abstract double[] calculate(double[] weight);

	public String toString()
	{
		return this.getClass().getSimpleName();
	}
}
