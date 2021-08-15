/**
 */
package slco2nqc.tests;

import junit.framework.TestCase;

import junit.textui.TestRunner;

import slco2nqc.Signal2Motor;
import slco2nqc.Slco2nqcFactory;

/**
 * <!-- begin-user-doc -->
 * A test case for the model object '<em><b>Signal2 Motor</b></em>'.
 * <!-- end-user-doc -->
 * @generated
 */
public class Signal2MotorTest extends TestCase {

	/**
	 * The fixture for this Signal2 Motor test case.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	protected Signal2Motor fixture = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public static void main(String[] args) {
		TestRunner.run(Signal2MotorTest.class);
	}

	/**
	 * Constructs a new Signal2 Motor test case with the given name.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public Signal2MotorTest(String name) {
		super(name);
	}

	/**
	 * Sets the fixture for this Signal2 Motor test case.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	protected void setFixture(Signal2Motor fixture) {
		this.fixture = fixture;
	}

	/**
	 * Returns the fixture for this Signal2 Motor test case.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	protected Signal2Motor getFixture() {
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
		setFixture(Slco2nqcFactory.eINSTANCE.createSignal2Motor());
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

} //Signal2MotorTest
