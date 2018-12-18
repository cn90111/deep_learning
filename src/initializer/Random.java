package initializer;

public class Random extends AbstractInitializer
{
	@Override
	public double[] initialize(int length)
	{
		result = new double[length];
		for (int i = 0; i < length; i++)
		{
			result[i] = Math.random();
		}
		return result;
	}
}
