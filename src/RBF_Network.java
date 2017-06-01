import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import org.apache.commons.math3.*;
import org.apache.commons.math3.stat.descriptive.moment.StandardDeviation;

public class RBF_Network {
	
	private static double [][] entradas;
	private static double [][] saidas;
	
	private static double bias1[];
	private static double bias2[];
	
	private static double antigodeltabs1[];
	private static double antigodeltabs2[];
	private static double deltabs1[];
	private static double deltabs2[];
	
	private static double net1[];
	private static double net2[][];
	private static double netout[][];
	private static double erro1[][];
	private static double erro2[][];

	private static double sensibilidade1[];

	private static double fnet1[];
	
	private static int numHidden;
	private static int numInputs;
	private static int numOutputs;
	private static int ciclos;
	private static int cicloAtual;
	private static int numTraining;
	//Atributo que indica o valor de Lcoef (taxa de aprendizagem).
	private static double alfa;
	//Atributo que indica o valor de Motum (Momentum).
	private static double beta;
	
	private static double[] [] pesos1;
	private static double[] [] pesos2;
	
	private static double [][] antigodeltapesos1;
	private static double [][] antigodeltapesos2;

	private static double [][] deltapesos1;
	private static double [][] deltapesos2;
	
	static List<String> teste = new ArrayList<String>();
	static double[] valoresEntrada;	
	
	
	public RBF_Network(int ciclos, int numInput, int numHidden, int numOutput,
			int numTraining, double alpha, double beta){
		setCiclos(ciclos);
		setAlfa(alpha);
		setBeta(beta);
		setNumInputs(numInput);
		setNumHidden(numHidden);
		setNumOutputs(numOutput);
		setNumTraining(numTraining); //setar o numero de treino como 70% da base
		
		bias2 = new double[numOutput];
		antigodeltabs2 = new double[numOutput];
		deltabs2 = new double[10];
	}
	
	public void Busca(){
		pesos1 = initPesos(getNumHidden(), getNumInputs());
		antigodeltapesos1 = initPesos(getNumHidden(), getNumInputs());
		deltapesos1 = initPesos(getNumHidden(), getNumInputs());
		
		pesos2 = initPesos(getNumOutputs(), getNumHidden());
		antigodeltapesos2 = initPesos(getNumOutputs(), getNumHidden());
		deltapesos2 = initPesos(getNumOutputs(), getNumHidden());
		
		net2 = initPesos(getNumTraining(), getNumHidden());
		netout = initPesos(getNumTraining(), getNumHidden());
		erro2 = initPesos(getNumTraining(), getNumHidden());
		
		erro1 = initPesos(getNumTraining(), getNumHidden());
		
		randomize();
		
		
		teste = (ArrayList<String>) readFile("C:\\Users\\Mariana\\Documents\\Poli\\8º período\\Computação Natural\\Evolução natural\\Bases\\Bases\\sunspot.txt");
		
		
		//double [][] entradasIniciais = {{5,11,16}, {11,16,23}, {16,23,36}
		//,{23, 36, 58}, {36,58,29}};
		entradas = new double [teste.size()][numInputs];

		//double [][] saidasIniciais = {{23}, {36},{58},{29},{20}};
		saidas = new double [teste.size()][1];
		
		normaliza();
		organizaEntradaESaida();
		forward();
		
		
	}
	
	public static void randomize(){
		Random random = new Random();
		for (int j=0; j<getNumHidden(); j++) {
			setBias1(j,-1+random.nextInt(8192)/8192);
			setAntigodeltabs1(j, 0.0d);
			setDeltabs1(j, 0.0d);
			
			for (int i=0; i<getNumInputs(); i++) {
				setPesos1(j, i, random.nextInt(8192)/8192-0.5d);
				setAntigodeltapesos1(j, i, 0.0d);
				setDeltapesos1(j, i, 0.0d);
			}
		}
		
		for (int j=0; j<getNumOutputs(); j++) {
			bias2[j]= -0.1 +(random.nextInt((8192)/8192));
			setAntigodeltabs2(j, 0.0d);
			for (int i=0;i<getNumHidden(); i++) {
				setPesos2(j, i, 0.1d * random.nextInt(8192)/8192- 0.05);
				setAntigodeltapesos2(j, i, 0.0d);
				setDeltapesos2(j, i, 0.0d);
			}
		}
	}
	
	public static void forward(){
		for(int kl=0; kl<getCiclos(); kl++) {
			setCicloAtual(getCicloAtual()+ 1);
			
			for(int itr=0; itr<getNumTraining(); itr++) {

				double ea,eb;
				for (int j=0;j<getNumHidden();j++) {
					net1[j] = getBias1()[j];
					for(int i=0;i<getNumInputs();i++){
						net1[j] = (getNet1()[j]+ (getPesos1()[j][i]* getEntradas()[itr][i]));
					}

					ea=(double)(Math.exp((double)((-1.0d)*(net1[j]))));

					fnet1[j] = ((double) (1.0)/(1.0 + (ea)));
				}
	
				for(int j=0;j<getNumOutputs();j++)
				{
					net2[itr][j]= bias2[j];
					for(int i=0;i<getNumHidden();i++){
						net2[itr][j]= (net2[itr][j]+(pesos2[j][i]*fnet1[i]));
					}
					eb=(double)(Math.exp((double)((-1.0d)*net2[itr][j])));
					
					netout[itr][j] = (double) (1.0/(1.0+eb));
				}

				//Reajustando os pesos
				for(int j=0;j<getNumOutputs();j++) {
					erro2 [itr][j] = (saidas[itr][j] - netout[itr][j]);
					//impressão dos dados de saída
					System.out.println("Ciclo:"+ " "+ getCicloAtual() + "  "+	"Exemplo:" +" "+ (itr+1));

					System.out.println("Saída desejada:"+" "  +  saidas[itr][j]+ "  "+ "Saída calculada:" 
					+" " +netout[itr][j]);
					System.out.println("Erro:" +" "+ getErro2()[itr][j]);
					
									
					setDeltabs2(j,( getAlfa() * getErro2()[itr][j]* netout[itr][j])*
					(1.0-getNetout()[itr][j])+(getBeta()) * antigodeltabs2[j]);

					for(int i=0;i<getNumHidden();i++){
						setDeltapesos2(j, i,( getAlfa()*erro2[itr][j])*
								netout[itr][j]*(1.0-getNetout()[itr][j])*fnet1[i]+
								(getBeta()*antigodeltapesos2[j][i]));
					}
				}

				for(int j=0;j<getNumHidden();j++) {
					sensibilidade1[j] =  0.0d;
					for(int i=0;i<getNumOutputs();i++) {
						sensibilidade1[j]= getSensibilidade1()[j]+(erro2[itr][i])*getPesos2()[i][j];
					}

					erro1[itr][j] = (fnet1[j])*(1.0d-fnet1[j])*(sensibilidade1[j]);
					setDeltabs1(j,( getAlfa() * erro1[itr][j])+
							(getBeta() * antigodeltabs1[j]));

					for(int ii=0;ii<getNumInputs();ii++) {
						setDeltapesos1(j,ii, (getAlfa() * erro1[itr][j])*(entradas[itr][ii])+
								(getBeta() * antigodeltapesos1[j][ii]));
					}
				}

				for(int j=0;j<getNumHidden();j++) {
					setBias1(j, deltabs1[j] + bias1[j]);
					setAntigodeltabs1(j, deltabs1[j]);
					//System.out.println("bias:"+" " + (j+1) +"     "+ bias1[j]+ "  ");

					for(int ii=0;ii<getNumInputs();ii++) {

						setPesos1(j,ii, pesos1[j][ii]+deltapesos1[j][ii]);
						//System.out.println("Peso:" +" "+ (j+1)+" " + (ii+1) + "    " +pesos1[j][ii]);

						setAntigodeltapesos1(j,ii, deltapesos1[j][ii]);
					}
				}

				for(int j=0;j<getNumOutputs();j++) {
					bias2[j] =  deltabs2[j] + bias2[j];
					setAntigodeltabs2(j, deltabs2[j]);
					//System.out.println("bias:"+" " + (j+1) +"     "+ bias2[j]+ "  ");

					for(int i=0;i<getNumHidden();i++) {
						pesos2[j][i] =  pesos2[j][i]+ deltapesos2[j][i];
						setAntigodeltapesos2(j, i, deltapesos2[j][i]);
						//System.out.println("Peso:" +" "+ (j+1)+" " + (i+1) + "    " +pesos2[j][i]);

						
					}
				}
			}
		}
	}
	
	//método para inicializar os pesos 
	
	public static double[][] getEntradas() {
		return entradas;
	}

	public static void setEntradas(double[][] entradas) {
		RBF_Network.entradas = entradas;
	}

	public static double[][] getSaidas() {
		return saidas;
	}

	public static void setSaidas(double[][] saidas) {
		RBF_Network.saidas = saidas;
	}

	public static double[] getBias1() {
		return bias1;
	}

	public static void setBias1(int index, double value) {
		bias1[index] = value;
	}

	public static double[] getBias2() {
		return bias2;
	}

	public static void setBias2(int index, double value) {
		bias2[index] = value;
	}

	public static double[] getAntigodeltabs1() {
		return antigodeltabs1;
	}

	public static void setAntigodeltabs1(int index, double value) {
		antigodeltabs1[index] = value;
	}

	public static double[] getAntigodeltabs2() {
		return antigodeltabs2;
	}

	public static void setAntigodeltabs2(int index, double value) {
		antigodeltabs2[index] = value;
	}

	public static double[] getDeltabs1() {
		return deltabs1;
	}

	public static void setDeltabs1(int index, double value) {
		deltabs1[index] = value;
	}

	public static double[] getDeltabs2() {
		return deltabs2;
	}

	public static void setDeltabs2(int index, double value) {
		deltabs2[index] = value;
	}

	public static double[] getNet1() {
		return net1;
	}

	public static void setNet1(double[] net1) {
		RBF_Network.net1 = net1;
	}

	public static double[][] getNet2() {
		return net2;
	}

	public static void setNet2(double[][] net2) {
		RBF_Network.net2 = net2;
	}

	public static double[][] getNetout() {
		return netout;
	}

	public static void setNetout(double[][] netout) {
		RBF_Network.netout = netout;
	}

	public static double[][] getErro1() {
		return erro1;
	}

	public static void setErro1(double[][] erro1) {
		RBF_Network.erro1 = erro1;
	}

	public static double[][] getErro2() {
		return erro2;
	}

	public static void setErro2(double[][] erro2) {
		RBF_Network.erro2 = erro2;
	}

	public static double[] getSensibilidade1() {
		return sensibilidade1;
	}

	public static void setSensibilidade1(double[] sensibilidade1) {
		RBF_Network.sensibilidade1 = sensibilidade1;
	}

	public static double[] getFnet1() {
		return fnet1;
	}

	public static void setFnet1(double[] fnet1) {
		RBF_Network.fnet1 = fnet1;
	}

	public static double[][] getPesos1() {
		return pesos1;
	}

	public static void setPesos1(int row, int col, double value) {
		pesos1[row][col] = value;
	}

	public static double[][] getPesos2() {
		return pesos2;
	}

	public static void setPesos2(int row, int col, double value) {
		pesos2[row][col] = value;
	}

	public static double[][] getAntigodeltapesos1() {
		return antigodeltapesos1;
	}

	public static void setAntigodeltapesos1(int row, int col, double value) {
		antigodeltapesos1[row][col] = value;
	}

	public static double[][] getAntigodeltapesos2() {
		return antigodeltapesos2;
	}

	public static void setAntigodeltapesos2(int row, int col, double value) {
		antigodeltapesos2[row][col] = value;
	}

	public static double[][] getDeltapesos1() {
		return deltapesos1;
	}

	public static void setDeltapesos1(int row, int col, double value) {
		deltapesos1[row][col] = value;
	}

	public static double[][] getDeltapesos2() {
		return deltapesos2;
	}

	public static void setDeltapesos2(int row, int col, double value) {
		deltapesos2[row][col] = value;
	}

	public static double[][] initPesos(int row, int col){
		double [] [] aux = new double[row][col];
		
		return aux;
	}
	
	//Get e set
	public static int getNumHidden() {
		return numHidden;
	}
	public static void setNumHidden(int numHidden) {
		RBF_Network.numHidden = numHidden;
		
		//Inicializa o vetor para bias1, netin1, sum1, prdlbs1, delbs1.
				bias1 = new double[getNumHidden()];
				net1 = new double[getNumHidden()];
				sensibilidade1 = new double[getNumHidden()];
				fnet1 = new double[getNumHidden()];
				antigodeltabs1 = new double [getNumHidden()];
				deltabs1 = new double [getNumHidden()];
	}
	public static int getNumInputs() {
		return numInputs;
	}
	public static void setNumInputs(int numInputs) {
		RBF_Network.numInputs = numInputs;
	}
	public static int getNumOutputs() {
		return numOutputs;
	}
	public static void setNumOutputs(int numOutputs) {
		RBF_Network.numOutputs = numOutputs;
	}
	public static int getCiclos() {
		return ciclos;
	}
	public static void setCiclos(int ciclos) {
		RBF_Network.ciclos = ciclos;
	}
	public static int getCicloAtual() {
		return cicloAtual;
	}
	public static void setCicloAtual(int cicloAtual) {
		RBF_Network.cicloAtual = cicloAtual;
	}
	public static int getNumTraining() {
		return numTraining;
	}
	public static void setNumTraining(int numTraining) {
		RBF_Network.numTraining = numTraining;
	}
	public static double getAlfa() {
		return alfa;
	}
	public static void setAlfa(double alfa) {
		RBF_Network.alfa = alfa;
	}
	public static double getBeta() {
		return beta;
	}
	public static void setBeta(double beta) {
		RBF_Network.beta = beta;
	}
	
	/**
	 * Open and read a file, and return the lines in the file as a list
	 * of Strings.
	 * (Demonstrates Java FileReader, BufferedReader, and Java5.)
	 */
	public List<String> readFile(String filename)
	{
	  List<String> records = new ArrayList<String>();
	  try
	  {
	    BufferedReader reader = new BufferedReader(new FileReader(filename));
	    String line;
	    while ((line = reader.readLine()) != null)
	    {
	      records.add(line);
	    }
	    reader.close();
	    teste = records;
	    return records;
	  }
	  catch (Exception e)
	  {
	    System.err.format("Exception occurred trying to read '%s'.", filename);
	    e.printStackTrace();
	    return null;
	  }
	}
	
	//Método para normalizar os valores entre [0,1]
	
	public void normaliza(){
		
		StandardDeviation sd = new StandardDeviation();
		
		//valor = (valor - media)/desvio padrão
		double max, min;
		double media = 0;
		valoresEntrada = new double[teste.size()];
		double[] auxEntrada = new double[teste.size()];
		
		max = min = Double.parseDouble(teste.get(0));
		
		for(int  i = 0; i < teste.size(); i++){
			valoresEntrada[i] = Double.parseDouble(teste.get(i));
			media = media + valoresEntrada[i];
			
			if(valoresEntrada[i] > max){
				max = valoresEntrada[i];
			}
			if(valoresEntrada[i] < min){
				min = valoresEntrada[i];
		}
			
		}
		auxEntrada = valoresEntrada;
		media = media/valoresEntrada.length;
		
		for(int j = 0; j < valoresEntrada.length; j++){
			valoresEntrada[j] = (auxEntrada[j] - media)/sd.evaluate(auxEntrada);
		}
			
	}
	
	public void organizaEntradaESaida(){
		int m = numInputs - 1;
		int j;
		double[] aux = new double[getNumInputs()];
			for(int i = 0; i < teste.size() - m; i++){
				for(j = 0; j < getNumInputs() - m; j++){
					entradas[i][j] = valoresEntrada[i+j];
				}
				saidas[i][0] = valoresEntrada[i+m];
			}
		
	}

}
