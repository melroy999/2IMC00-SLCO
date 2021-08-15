/**
 */
package slco2nqc.tests;

import junit.framework.TestCase;

import junit.textui.TestRunner;

import slco2nqc.Signal2Sensor;
import slco2nqc.Slco2nqcFactory;

/**
 * <!-- begin-user-doc -->
 * A test case for the model object '<em><b>Signal2 Sensor</b></em>'.
 * <!-- end-user-doc -->
 * @generated
 */
public class Signal2SensorTest extends TestCase {

	/**
	 * The fixture for this Signal2 Sensor test case.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	protected Signal2Sensor fixture = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public static void main(String[] args) {
		TestRunner.run(Signal2SensorTest.class);
	}

	/**
	 * Constructs a new Signal2 Sensor test case with the given name.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public Signal2SensorTest(String name) {
		super(name);
	}

	/**
	 * Sets the fixture for this Signal2 Sensor test case.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	protected void setFixture(Signal2Sensor fixture) {
		this.fixture = fixture;
	}

	/**
	 * Returns the fixture for this Signal2 Sensor test case.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	protected Signal2Sensor getFixture() {
		return fixture;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see junit.framework.TestCase#setUp()
	 * @generated
	 */
	@Override
	protected void setUp() throws Exception {
		setFixture(Slco2nqcFactory.eINSTANCE.createSignal2Sensor());
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see junit.framework.TestCase#tearDown()
	 * @generated
	 */
	@Override
	protected void tearDown() throws Exception {
		setFixture(null);
	}

} //Signal2SensorTest
