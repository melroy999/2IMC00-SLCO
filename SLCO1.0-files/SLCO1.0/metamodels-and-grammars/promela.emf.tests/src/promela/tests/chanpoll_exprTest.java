/**
 */
package promela.tests;

import junit.textui.TestRunner;

import promela.PromelaFactory;
import promela.chanpoll_expr;

/**
 * <!-- begin-user-doc -->
 * A test case for the model object '<em><b>chanpoll expr</b></em>'.
 * <!-- end-user-doc -->
 * @generated
 */
public class chanpoll_exprTest extends exprTest {

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public static void main(String[] args) {
		TestRunner.run(chanpoll_exprTest.class);
	}

	/**
	 * Constructs a new chanpoll expr test case with the given name.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public chanpoll_exprTest(String name) {
		super(name);
	}

	/**
	 * Returns the fixture for this chanpoll expr test case.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	protected chanpoll_expr getFixture() {
		return (chanpoll_expr)fixture;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see junit.framework.TestCase#setUp()
	 * @generated
	 */
	@Override
	protected void setUp() throws Exception {
		setFixture(PromelaFactory.eINSTANCE.createchanpoll_expr());
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

} //chanpoll_exprTest
