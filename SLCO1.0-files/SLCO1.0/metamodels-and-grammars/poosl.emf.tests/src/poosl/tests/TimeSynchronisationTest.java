/**
 */
package poosl.tests;

import junit.textui.TestRunner;

import poosl.PooslFactory;
import poosl.TimeSynchronisation;

/**
 * <!-- begin-user-doc -->
 * A test case for the model object '<em><b>Time Synchronisation</b></em>'.
 * <!-- end-user-doc -->
 * @generated
 */
public class TimeSynchronisationTest extends StatementTest {

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public static void main(String[] args) {
		TestRunner.run(TimeSynchronisationTest.class);
	}

	/**
	 * Constructs a new Time Synchronisation test case with the given name.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public TimeSynchronisationTest(String name) {
		super(name);
	}

	/**
	 * Returns the fixture for this Time Synchronisation test case.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	protected TimeSynchronisation getFixture() {
		return (TimeSynchronisation)fixture;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see junit.framework.TestCase#setUp()
	 * @generated
	 */
	@Override
	protected void setUp() throws Exception {
		setFixture(PooslFactory.eINSTANCE.createTimeSynchronisation());
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

} //TimeSynchronisationTest
