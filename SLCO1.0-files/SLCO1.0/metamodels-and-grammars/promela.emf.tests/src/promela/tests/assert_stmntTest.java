/**
 */
package promela.tests;

import junit.textui.TestRunner;

import promela.PromelaFactory;
import promela.assert_stmnt;

/**
 * <!-- begin-user-doc -->
 * A test case for the model object '<em><b>assert stmnt</b></em>'.
 * <!-- end-user-doc -->
 * @generated
 */
public class assert_stmntTest extends stmntTest {

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public static void main(String[] args) {
		TestRunner.run(assert_stmntTest.class);
	}

	/**
	 * Constructs a new assert stmnt test case with the given name.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public assert_stmntTest(String name) {
		super(name);
	}

	/**
	 * Returns the fixture for this assert stmnt test case.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	protected assert_stmnt getFixture() {
		return (assert_stmnt)fixture;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see junit.framework.TestCase#setUp()
	 * @generated
	 */
	@Override
	protected void setUp() throws Exception {
		setFixture(PromelaFactory.eINSTANCE.createassert_stmnt());
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

} //assert_stmntTest
