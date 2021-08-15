/**
 */
package promela.tests;

import junit.textui.TestRunner;

import promela.PromelaFactory;
import promela.name_expr;

/**
 * <!-- begin-user-doc -->
 * A test case for the model object '<em><b>name expr</b></em>'.
 * <!-- end-user-doc -->
 * @generated
 */
public class name_exprTest extends any_exprTest {

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public static void main(String[] args) {
		TestRunner.run(name_exprTest.class);
	}

	/**
	 * Constructs a new name expr test case with the given name.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public name_exprTest(String name) {
		super(name);
	}

	/**
	 * Returns the fixture for this name expr test case.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	protected name_expr getFixture() {
		return (name_expr)fixture;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see junit.framework.TestCase#setUp()
	 * @generated
	 */
	@Override
	protected void setUp() throws Exception {
		setFixture(PromelaFactory.eINSTANCE.createname_expr());
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

} //name_exprTest
