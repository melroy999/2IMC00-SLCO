/**
 */
package slco2nqc.tests;

import junit.framework.TestCase;

import junit.textui.TestRunner;

import slco2nqc.MotorInit;
import slco2nqc.Slco2nqcFactory;

/**
 * <!-- begin-user-doc -->
 * A test case for the model object '<em><b>Motor Init</b></em>'.
 * <!-- end-user-doc -->
 * @generated
 */
public class MotorInitTest extends TestCase {

	/**
	 * The fixture for this Motor Init test case.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	protected MotorInit fixture = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public static void main(String[] args) {
		TestRunner.run(MotorInitTest.class);
	}

	/**
	 * Constructs a new Motor Init test case with the given name.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public MotorInitTest(String name) {
		super(name);
	}

	/**
	 * Sets the fixture for this Motor Init test case.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	protected void setFixture(MotorInit fixture) {
		this.fixture = fixture;
	}

	/**
	 * Returns the fixture for this Motor Init test case.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	protected MotorInit getFixture() {
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
		setFixture(Slco2nqcFactory.eINSTANCE.createMotorInit());
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

} //MotorInitTest
