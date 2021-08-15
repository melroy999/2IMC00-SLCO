/**
 */
package poosl.tests;

import junit.textui.TestRunner;

import poosl.PooslFactory;
import poosl.Skip;

/**
 * <!-- begin-user-doc -->
 * A test case for the model object '<em><b>Skip</b></em>'.
 * <!-- end-user-doc -->
 * @generated
 */
public class SkipTest extends StatementTest {

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public static void main(String[] args) {
		TestRunner.run(SkipTest.class);
	}

	/**
	 * Constructs a new Skip test case with the given name.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public SkipTest(String name) {
		super(name);
	}

	/**
	 * Returns the fixture for this Skip test case.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	protected Skip getFixture() {
		return (Skip)fixture;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see junit.framework.TestCase#setUp()
	 * @generated
	 */
	@Override
	protected void setUp() throws Exception {
		setFixture(PooslFactory.eINSTANCE.createSkip());
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

} //SkipTest
