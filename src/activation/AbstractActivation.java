package activation;

public abstract class AbstractActivation
{
	private double[] result;

	public AbstractActivation()
	{
	}

	public final double[] getError(double[] data)
	{
		for (int i = 0; i < data.length; i++)
		{
			result[i] = calculate(data[i]);
		}
		return result;
	}

	protected abstract double calculate(double data);
}
