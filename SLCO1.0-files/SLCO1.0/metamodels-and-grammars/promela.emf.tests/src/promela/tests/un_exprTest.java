/**
 */
package promela.tests;

import junit.textui.TestRunner;

import promela.PromelaFactory;
import promela.un_expr;

/**
 * <!-- begin-user-doc -->
 * A test case for the model object '<em><b>un expr</b></em>'.
 * <!-- end-user-doc -->
 * @generated
 */
public class un_exprTest extends any_exprTest {

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public static void main(String[] args) {
		TestRunner.run(un_exprTest.class);
	}

	/**
	 * Constructs a new un expr test case with the given name.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public un_exprTest(String name) {
		super(name);
	}

	/**
	 * Returns the fixture for this un expr test case.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	protected un_expr getFixture() {
		return (un_expr)fixture;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see junit.framework.TestCase#setUp()
	 * @generated
	 */
	@Override
	protected void setUp() throws Exception {
		setFixture(PromelaFactory.eINSTANCE.createun_expr());
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

} //un_exprTest
