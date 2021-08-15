/**
 */
package nqc.tests;

import junit.textui.TestRunner;

import nqc.BlockStatement;
import nqc.NqcFactory;

/**
 * <!-- begin-user-doc -->
 * A test case for the model object '<em><b>Block Statement</b></em>'.
 * <!-- end-user-doc -->
 * @generated
 */
public class BlockStatementTest extends StatementTest {

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public static void main(String[] args) {
		TestRunner.run(BlockStatementTest.class);
	}

	/**
	 * Constructs a new Block Statement test case with the given name.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public BlockStatementTest(String name) {
		super(name);
	}

	/**
	 * Returns the fixture for this Block Statement test case.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	protected BlockStatement getFixture() {
		return (BlockStatement)fixture;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see junit.framework.TestCase#setUp()
	 * @generated
	 */
	@Override
	protected void setUp() throws Exception {
		setFixture(NqcFactory.eINSTANCE.createBlockStatement());
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

} //BlockStatementTest
