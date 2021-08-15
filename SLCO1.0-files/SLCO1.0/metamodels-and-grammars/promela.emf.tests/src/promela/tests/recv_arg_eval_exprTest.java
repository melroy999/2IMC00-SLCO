/**
 */
package promela.tests;

import junit.textui.TestRunner;

import promela.PromelaFactory;
import promela.recv_arg_eval_expr;

/**
 * <!-- begin-user-doc -->
 * A test case for the model object '<em><b>recv arg eval expr</b></em>'.
 * <!-- end-user-doc -->
 * @generated
 */
public class recv_arg_eval_exprTest extends recv_argTest {

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public static void main(String[] args) {
		TestRunner.run(recv_arg_eval_exprTest.class);
	}

	/**
	 * Constructs a new recv arg eval expr test case with the given name.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public recv_arg_eval_exprTest(String name) {
		super(name);
	}

	/**
	 * Returns the fixture for this recv arg eval expr test case.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	protected recv_arg_eval_expr getFixture() {
		return (recv_arg_eval_expr)fixture;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see junit.framework.TestCase#setUp()
	 * @generated
	 */
	@Override
	protected void setUp() throws Exception {
		setFixture(PromelaFactory.eINSTANCE.createrecv_arg_eval_expr());
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

} //recv_arg_eval_exprTest
