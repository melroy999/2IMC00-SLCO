/**
 */
package slco2nqc.tests;

import junit.framework.TestCase;

import junit.textui.TestRunner;

import slco2nqc.SensorInitList;
import slco2nqc.Slco2nqcFactory;

/**
 * <!-- begin-user-doc -->
 * A test case for the model object '<em><b>Sensor Init List</b></em>'.
 * <!-- end-user-doc -->
 * @generated
 */
public class SensorInitListTest extends TestCase {

	/**
	 * The fixture for this Sensor Init List test case.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	protected SensorInitList fixture = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public static void main(String[] args) {
		TestRunner.run(SensorInitListTest.class);
	}

	/**
	 * Constructs a new Sensor Init List test case with the given name.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public SensorInitListTest(String name) {
		super(name);
	}

	/**
	 * Sets the fixture for this Sensor Init List test case.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	protected void setFixture(SensorInitList fixture) {
		this.fixture = fixture;
	}

	/**
	 * Returns the fixture for this Sensor Init List test case.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	protected SensorInitList getFixture() {
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
		setFixture(Slco2nqcFactory.eINSTANCE.createSensorInitList());
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

} //SensorInitListTest
