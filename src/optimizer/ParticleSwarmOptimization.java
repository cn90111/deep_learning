package optimizer;

public class ParticleSwarmOptimization extends AbstractOptimizer
{
	private int particleSize;
	private double c1;
	private double c2;

	public ParticleSwarmOptimization(int particleSize, double globalSearchWeight, double localSearchWeight)
	{
		this.particleSize = particleSize;
		this.c1 = localSearchWeight;
		this.c2 = globalSearchWeight;
	}

	@Override
	public double[][][] updateWeight(double error)
	{
		// TODO Auto-generated method stub
		return null;
	}
}
