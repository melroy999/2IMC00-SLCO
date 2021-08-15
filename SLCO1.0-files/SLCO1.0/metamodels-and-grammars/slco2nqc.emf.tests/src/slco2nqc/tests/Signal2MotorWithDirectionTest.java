/**
 */
package slco2nqc.tests;

import junit.textui.TestRunner;

import slco2nqc.Signal2MotorWithDirection;
import slco2nqc.Slco2nqcFactory;

/**
 * <!-- begin-user-doc -->
 * A test case for the model object '<em><b>Signal2 Motor With Direction</b></em>'.
 * <!-- end-user-doc -->
 * @generated
 */
public class Signal2MotorWithDirectionTest extends Signal2MotorTest {

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public static void main(String[] args) {
		TestRunner.run(Signal2MotorWithDirectionTest.class);
	}

	/**
	 * Constructs a new Signal2 Motor With Direction test case with the given name.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public Signal2MotorWithDirectionTest(String name) {
		super(name);
	}

	/**
	 * Returns the fixture for this Signal2 Motor With Direction test case.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	protected Signal2MotorWithDirection getFixture() {
		return (Signal2MotorWithDirection)fixture;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see junit.framework.TestCase#setUp()
	 * @generated
	 */
	@Override
	protected void setUp() throws Exception {
		setFixture(Slco2nqcFactory.eINSTANCE.createSignal2MotorWithDirection());
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

} //Signal2MotorWithDirectionTest
