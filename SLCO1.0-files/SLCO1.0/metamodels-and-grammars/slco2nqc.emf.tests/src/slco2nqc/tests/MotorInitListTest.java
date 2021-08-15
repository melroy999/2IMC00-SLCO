/**
 */
package slco2nqc.tests;

import junit.framework.TestCase;

import junit.textui.TestRunner;

import slco2nqc.MotorInitList;
import slco2nqc.Slco2nqcFactory;

/**
 * <!-- begin-user-doc -->
 * A test case for the model object '<em><b>Motor Init List</b></em>'.
 * <!-- end-user-doc -->
 * @generated
 */
public class MotorInitListTest extends TestCase {

	/**
	 * The fixture for this Motor Init List test case.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	protected MotorInitList fixture = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public static void main(String[] args) {
		TestRunner.run(MotorInitListTest.class);
	}

	/**
	 * Constructs a new Motor Init List test case with the given name.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public MotorInitListTest(String name) {
		super(name);
	}

	/**
	 * Sets the fixture for this Motor Init List test case.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	protected void setFixture(MotorInitList fixture) {
		this.fixture = fixture;
	}

	/**
	 * Returns the fixture for this Motor Init List test case.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	protected MotorInitList getFixture() {
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
		setFixture(Slco2nqcFactory.eINSTANCE.createMotorInitList());
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

} //MotorInitListTest
