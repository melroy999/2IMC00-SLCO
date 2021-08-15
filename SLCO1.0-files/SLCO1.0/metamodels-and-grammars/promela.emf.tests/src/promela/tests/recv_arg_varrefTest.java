/**
 */
package promela.tests;

import junit.textui.TestRunner;

import promela.PromelaFactory;
import promela.recv_arg_varref;

/**
 * <!-- begin-user-doc -->
 * A test case for the model object '<em><b>recv arg varref</b></em>'.
 * <!-- end-user-doc -->
 * @generated
 */
public class recv_arg_varrefTest extends recv_argTest {

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public static void main(String[] args) {
		TestRunner.run(recv_arg_varrefTest.class);
	}

	/**
	 * Constructs a new recv arg varref test case with the given name.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public recv_arg_varrefTest(String name) {
		super(name);
	}

	/**
	 * Returns the fixture for this recv arg varref test case.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	protected recv_arg_varref getFixture() {
		return (recv_arg_varref)fixture;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see junit.framework.TestCase#setUp()
	 * @generated
	 */
	@Override
	protected void setUp() throws Exception {
		setFixture(PromelaFactory.eINSTANCE.createrecv_arg_varref());
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

} //recv_arg_varrefTest
