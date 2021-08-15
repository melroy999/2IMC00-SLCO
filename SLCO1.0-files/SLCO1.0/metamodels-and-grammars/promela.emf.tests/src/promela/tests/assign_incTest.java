/**
 */
package promela.tests;

import junit.textui.TestRunner;

import promela.PromelaFactory;
import promela.assign_inc;

/**
 * <!-- begin-user-doc -->
 * A test case for the model object '<em><b>assign inc</b></em>'.
 * <!-- end-user-doc -->
 * @generated
 */
public class assign_incTest extends assignTest {

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public static void main(String[] args) {
		TestRunner.run(assign_incTest.class);
	}

	/**
	 * Constructs a new assign inc test case with the given name.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public assign_incTest(String name) {
		super(name);
	}

	/**
	 * Returns the fixture for this assign inc test case.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	protected assign_inc getFixture() {
		return (assign_inc)fixture;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see junit.framework.TestCase#setUp()
	 * @generated
	 */
	@Override
	protected void setUp() throws Exception {
		setFixture(PromelaFactory.eINSTANCE.createassign_inc());
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

} //assign_incTest
