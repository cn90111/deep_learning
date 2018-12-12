package initializer;

public abstract class AbstractInitializer
{
	protected double[] result;

	public AbstractInitializer()
	{
	}

	public abstract double[] initialize(int length);
}
